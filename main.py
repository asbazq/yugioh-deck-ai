# app/main.py
import os, uuid, shutil, tempfile
from typing import List, Optional

import cv2
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ====== 프로젝트 의존 코드(당신 저장소 기준 경로에 맞춤) ======
# models / data / utils 는 기존 레포에서 가져옵니다.
from models.torch import Detector
from data.preprocess.torch import detector_preprocessing
from utils.image_utils import make_square_shape

# ---- ChromaDB 연결 어댑터 (간단 래퍼) ----
import chromadb


class ChromaDBConnection:
    def __init__(self):
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        collection = os.getenv("CHROMA_COLLECTION", "cards")

        # HTTP 서버로 띄운 chroma에 접속 (chroma-server 사용 시)
        self.client = chromadb.HttpClient(host=host, port=port)
        self.col = self.client.get_or_create_collection(collection)

    def search_by_embed(self, emb: List[float], topk: int = 5):
        out = self.col.query(
            query_embeddings=[emb],
            n_results=topk,
            include=["metadatas", "distances", "documents", "embeddings", "uris", "data"]
        )
        # 반환 형태 평탄화
        results = []
        for i in range(len(out["ids"])):
            items = []
            for j in range(len(out["ids"][i])):
                meta = out["metadatas"][i][j] or {}
                items.append({
                    "id": meta.get("id") or meta.get("ygopro_id"),
                    "name": meta.get("name") or meta.get("kor_name") or "",
                    "distance": (out["distances"][i][j] if out.get("distances") else None),
                })
            results.append(items)
        return results
# ============================================================


# ---------- 설정 ----------
MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.pt")
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))
TOPK = int(os.getenv("TOPK", "5"))
ALLOW_CORS = os.getenv("ALLOW_CORS", "1") == "1"

# ---------- 앱 ----------
app = FastAPI(title="YuGiOh AI Inference", version="1.0")

if ALLOW_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 내부통신만이면 스프링에서 프록시할 거라 제거해도 OK
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ---------- 디바이스/모델/DB ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Detector()
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.to(device)
model.eval()

chroma_db = ChromaDBConnection()


# ---------- Pydantic 스키마 ----------
class DetOneBox(BaseModel):
    box: List[float]                 # [x1,y1,x2,y2]
    topk_names: List[str]
    topk_ids: List[str]


class PredictItem(BaseModel):
    boxes: List[List[float]]         # Nx4
    topk_names: List[List[str]]      # N x K
    topk_ids: List[List[str]]        # N x K
    result_image: Optional[str]      # /tmp/..png


class PredictResponse(BaseModel):
    results: List[PredictItem]


# ---------- 엔드포인트 ----------
@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)):
    # 1) 업로드 → 임시 파일
    suffix = os.path.splitext(image.filename or "")[-1] or ".jpg"
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{suffix}")

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # 2) 이미지 로드 (BGR→RGB)
    try:
        origin = cv2.imread(tmp_path)
        if origin is None:
            raise ValueError("cv2.imread failed")
        origin = origin[:, :, ::-1]
    finally:
        # 업로드 원본은 즉시 삭제
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # 3) 전처리
    square, _, _ = make_square_shape(origin, INPUT_SIZE)
    inputs = detector_preprocessing(square.copy())
    if len(inputs.shape) == 3:
        inputs = inputs[None, :]
    inputs = inputs.to(device)

    # 4) 추론
    with torch.no_grad():
        pred_det, pred_embed = model(inputs)
    input_shape = inputs.shape[2:4]
    det_results = model.postprocess(pred_det, pred_embed, input_shape, origin[None, :])

    # 5) 결과 조립 (+ 시각화 이미지 저장)
    outputs: List[PredictItem] = []
    for det in det_results:
        card_names = []
        card_ids = []
        for emb in det.embeds:
            # Chromadb top-k
            res = chroma_db.search_by_embed(emb.tolist(), TOPK)[0]
            names = [x.get("name", "") for x in res]
            ids = [str(x.get("id", "")) for x in res]
            card_names.append(names)
            card_ids.append(ids)

        det.names = card_names
        det.ids = card_ids

        out_path = os.path.join(tmp_dir, f"result_{uuid.uuid4().hex}.png")
        det.save(out_path)  # 기존 함수: 박스/라벨 덮어서 저장

        outputs.append(PredictItem(
            boxes=det.bboxes.tolist(),
            topk_names=card_names,
            topk_ids=card_ids,
            result_image=out_path
        ))

    return PredictResponse(results=outputs)


@app.get("/result_image")
def result_image(path: str = Query(..., description="absolute tmp path")):
    """ 프론트에서 /result_image?path=/tmp/result_x.png 로 땡겨가기 """
    if not path.startswith(tempfile.gettempdir()):
        raise HTTPException(status_code=400, detail="invalid path")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="image/png")
