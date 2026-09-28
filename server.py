import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, List

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from starlette.concurrency import run_in_threadpool

load_dotenv(".env")
logger = logging.getLogger(__name__)


def _getenv(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip('"').strip("'")


MODEL_PATH = _getenv("MODEL_PATH", "../best.pt")
TOP_K = int(_getenv("TOP_K", "5"))
EMBED_MODEL_VERSION = _getenv("EMBED_MODEL_VERSION", "")
EMBED_DIM = int(_getenv("EMBED_DIM", "0") or "0")
MAX_UPLOAD_BYTES = int(_getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(_getenv("MAX_IMAGE_PIXELS", "20000000"))
if not 1 <= TOP_K <= 100 or not 0 <= EMBED_DIM <= 4096:
    raise ValueError("TOP_K must be 1..100 and EMBED_DIM must be 0..4096")
if MAX_UPLOAD_BYTES <= 0 or MAX_IMAGE_PIXELS <= 0:
    raise ValueError("Upload and pixel limits must be positive")


class Candidate(BaseModel):
    id: str | int | None = None
    name: str | None = None


class DetectionResult(BaseModel):
    best: Candidate
    topk: List[Candidate]


class PredictResponse(BaseModel):
    detections: List[DetectionResult]
    elapsed: float


class EmbedDetection(BaseModel):
    bbox: List[float]
    embed: List[float]


class PredictEmbedsResponse(BaseModel):
    detections: List[EmbedDetection]
    elapsed: float


Embedding = Annotated[List[FiniteFloat], Field(min_length=1, max_length=4096)]


class SearchEmbedsRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    embeds: List[Embedding] = Field(min_length=1, max_length=128)
    top_k: int | None = Field(default=None, ge=1, le=100)
    model_version: str | None = None


class SearchEmbedsResponse(BaseModel):
    results: List[DetectionResult]
    elapsed: float


class TorchRuntime:
    """Load serving resources at startup, not when importing the API module."""

    def __init__(self):
        import torch
        from data.preprocess.torch import detector_preprocessing
        from utils.image_utils import make_square_shape
        from models.torch import Detector
        from db import ChromaDBConnection

        self.torch = torch
        self.preprocess = detector_preprocessing
        self.make_square = make_square_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = Detector()
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        self.detector.load_state_dict(state)
        self.detector.to(self.device)
        self.detector.eval()
        self.chroma = ChromaDBConnection()

    def infer(self, image):
        inputs, _, _ = self.make_square(image, 640)
        inputs = self.preprocess(inputs.copy())[None, :].to(self.device)
        with self.torch.inference_mode():
            pred_det, pred_embed = self.detector(inputs)
            return self.detector.postprocess(pred_det, pred_embed, inputs.shape[2:4], image[None, :])

    def search(self, embed, top_k):
        rows = self.chroma.search_by_embed(embed, n_result=top_k)
        return rows[0] if rows else []


def _bytes_to_rgb_ndarray(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=400, detail="Image is empty")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(status_code=400, detail="Only jpg/png/webp images are supported")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the pixel limit")
            return np.array(image.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="Image dimensions exceed the pixel limit") from exc


def _read_image(file: UploadFile):
    try:
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=400, detail="Only jpg/png/webp images are supported")
        raw = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image upload is too large")
        return _bytes_to_rgb_ndarray(raw)
    finally:
        file.file.close()


def _validate_embed_request(req: SearchEmbedsRequest):
    if EMBED_MODEL_VERSION and req.model_version != EMBED_MODEL_VERSION:
        raise HTTPException(status_code=400, detail="model_version mismatch or missing")
    dimension = EMBED_DIM or len(req.embeds[0])
    for idx, embed in enumerate(req.embeds):
        if len(embed) != dimension:
            raise HTTPException(status_code=400, detail=f"embed_dim mismatch at index {idx}")
        if not any(value != 0 for value in embed):
            raise HTTPException(status_code=400, detail=f"zero embedding at index {idx}")


def _search_result(runtime, embed, top_k):
    metas = runtime.search(embed, top_k) or []
    candidates = [Candidate(id=(m or {}).get("id"), name=(m or {}).get("name")) for m in metas]
    return DetectionResult(best=candidates[0] if candidates else Candidate(), topk=candidates)


def create_app(runtime_factory=TorchRuntime):
    @asynccontextmanager
    async def lifespan(application):
        application.state.runtime = await run_in_threadpool(runtime_factory)
        try:
            yield
        finally:
            application.state.runtime = None

    application = FastAPI(title="YuGiOh AI Recognizer", lifespan=lifespan)
    application.state.runtime = None
    inference_lock = threading.Lock()

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Raw NaN/Infinity values in a validation error cannot be encoded as JSON.
        return JSONResponse(status_code=422, content={"detail": [
            {key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()
        ]})
    application.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )

    def get_runtime(request):
        runtime = request.app.state.runtime
        if runtime is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return runtime

    def infer_file(request, file):
        # Reject excess GPU work instead of building an unbounded inference queue.
        if not inference_lock.acquire(blocking=False):
            file.file.close()
            raise HTTPException(status_code=503, detail="Inference is busy", headers={"Retry-After": "1"})
        try:
            return get_runtime(request).infer(_read_image(file))
        finally:
            inference_lock.release()

    # Synchronous endpoints use FastAPI's worker pool, keeping model/DB work off the event loop.
    @application.post("/predict", response_model=PredictResponse)
    def predict(request: Request, file: UploadFile = File(...)):
        start = time.perf_counter()
        try:
            runtime = get_runtime(request)
            detections = [_search_result(runtime, embed.tolist(), TOP_K)
                          for det in infer_file(request, file) for embed in det.embeds]
            return PredictResponse(detections=detections, elapsed=round(time.perf_counter() - start, 4))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed") from exc

    @application.post("/predict_embeds", response_model=PredictEmbedsResponse)
    def predict_embeds(request: Request, file: UploadFile = File(...)):
        start = time.perf_counter()
        try:
            detections = [EmbedDetection(bbox=bbox.tolist(), embed=embed.tolist())
                          for det in infer_file(request, file) for bbox, embed in zip(det.bboxes, det.embeds)]
            return PredictEmbedsResponse(detections=detections, elapsed=round(time.perf_counter() - start, 4))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed") from exc

    @application.post("/search_embeds", response_model=SearchEmbedsResponse)
    def search_embeds(req: SearchEmbedsRequest, request: Request):
        start = time.perf_counter()
        _validate_embed_request(req)
        try:
            runtime = get_runtime(request)
            results = [_search_result(runtime, embed, req.top_k or TOP_K) for embed in req.embeds]
            return SearchEmbedsResponse(results=results, elapsed=round(time.perf_counter() - start, 4))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Embedding search failed")
            raise HTTPException(status_code=500, detail="Search failed") from exc

    @application.get("/health")
    def health(request: Request):
        get_runtime(request)
        return {"ok": True}

    return application


app = create_app()
