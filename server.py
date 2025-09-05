# serve.py
import io
import os
import time
from typing import List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 프로젝트 유틸/모델
from data.preprocess.torch import detector_preprocessing
from utils.image_utils import make_square_shape
from models.torch import Detector

# DB 연결 래퍼
from db import ChromaDBConnection, MySQLConnection

# -----------------------------
# 환경 변수 로드
# -----------------------------
load_dotenv(".env")

def _getenv(key: str, default: str = "") -> str:
    v = os.getenv(key, default)
    # .env에 key="value" 형식이면 따옴표 제거
    return v.strip('"').strip("'")

MODEL_PATH = _getenv("MODEL_PATH", "../best.pt")   # 상위 폴더의 best.pt
TOP_K      = int(_getenv("TOP_K", "5"))

# -----------------------------
# FastAPI & CORS
# -----------------------------
app = FastAPI(title="YuGiOh AI Recognizer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 필요시 프론트 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 리소스 준비(모델/DB)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = Detector()
state = torch.load(MODEL_PATH, map_location="cpu")
detector.load_state_dict(state)
detector.to(device)
detector.eval()

chroma = ChromaDBConnection()    # .env의 host/chroma_* 사용
# mysql  = MySQLConnection()       # .env의 host/mysql_* 사용
mysql = None
def get_mysql():
    global mysql
    if mysql is None:
        from db import MySQLConnection
        try:
            mysql = MySQLConnection()
        except Exception as e:
            print("[WARN] MySQL unavailable:", e)
            # 사용처에서 None 체크
            return None
    return mysql

# -----------------------------
# 응답 스키마
# -----------------------------
class Candidate(BaseModel):
    id: str | int | None = None
    name: str | None = None

class DetectionResult(BaseModel):
    best: Candidate
    topk: List[Candidate]

class PredictResponse(BaseModel):
    detections: List[DetectionResult]
    elapsed: float

# -----------------------------
# 유틸
# -----------------------------
def _bytes_to_rgb_ndarray(b: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(b, np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("이미지 디코딩 실패")
    return bgr[:, :, ::-1]  # to RGB

# -----------------------------
# 라우트
# -----------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        # Only allow common image content types
        raise HTTPException(status_code=400, detail="Only jpg/png/webp images are supported")

    raw = await file.read()
    start = time.time()

    try:
        origin_img = _bytes_to_rgb_ndarray(raw)

        # 전처리
        student_inputs, _, _ = make_square_shape(origin_img, 640)
        student_inputs = detector_preprocessing(student_inputs.copy())
        student_inputs = student_inputs[None, :]  # (1, C, H, W)
        student_inputs = student_inputs.to(device)

        with torch.no_grad():
            pred_det, pred_embed = detector(student_inputs)

        input_shape = student_inputs.shape[2:4]
        det_results = detector.postprocess(
            pred_det, pred_embed, input_shape, origin_img[None, :]
        )

        detections: List[DetectionResult] = []

        # 감지된 카드 각각에 대해 벡터DB 검색
        for det in det_results:
            # det.embeds: (N, D)
            for embed in det.embeds:
                # ChromaDBConnection.search_by_embed는 metadatas를 반환
                metas: List[dict] = chroma.search_by_embed(embed.tolist(), n_result=TOP_K)[0]

                # top1
                top1_meta = metas[0] if metas else {}

                result = DetectionResult(
                    best=Candidate(
                        id=top1_meta.get("id"),
                        name=top1_meta.get("name"),
                    ),
                    topk=[
                        Candidate(
                            id=m.get("id"),
                            name=m.get("name"),
                        ) for m in metas
                    ]
                )
                detections.append(result)

        elapsed = time.time() - start
        return PredictResponse(detections=detections, elapsed=round(elapsed, 4))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e}")

# 간단 헬스체크
@app.get("/health")
def health():
    return {"ok": True}

