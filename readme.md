# Yu-Gi-Oh card finder

### Continuous evaluation → retraining

운영 모델을 정답 덱 이미지로 주기적으로 평가하고 인식 성능이 기준 아래일 때만
재학습합니다. 후보 검증, 중복 실행 방지, cooldown 및 실행 설정은
[continuous evaluation 안내](docs/continuous-evaluation.md)를 참조하세요.


Using yolov8 and EfficientNet, recognize Yu-Gi-Oh cards from the deck list image.


### Requirements
- numpy                        1.26.4
- opencv-python                4.10.0.84
- pandas                       2.2.2
- tensorflow                   2.15.0
- ultralytics                  8.2.82


### Motivation
Many people share their deck list by capturing game screens or another web page. However, putting all cards in one image, the card size will be reduced and it will be hard to recognize. If you want to know some cards in the shared deck list image, you have to ask the author what he used or manually compare 10000+ Yugioh cards. To solve this problem yolov8 is used to crop card images from the deck list, then embed each card image using EfficientNetB0, and find the card with the closest distance of the input image and the precomputed card vectors.


### Overall process
Deck list image → Yolov8 → card illustrations → Embedding model → calculate distance with card vectors → determine the card id


### Data preparation
All the card images for the dataset is downloaded from (ygoprodeck)[https://db.ygoprodeck.com]
To train embedding model, frist crop all illustrations from card images. The card itself contains lots of information about card types, text, level and atk/def points. However, the model couldn't focus on the illustration in the card, and thoes information distracts the model's training. The trained model without cropping determines the cards just by the color of the card border, rather than recognizing the illustration of the card.


### Data argumentation
Implemented zoom-in and zoom-out augmentation to make low pixel resolution of train image. The image size can be reduced up to -40% and the reduction ratio is randomly selected in every train steps.


### Used Model
EfficientNetB0 has been used for the backbone model and the Dense layers are removed. To train the model, the contrastive loss is used for the loss function. In each step, the model gets 3 inputs, the original image, positive image, and negative image. The original image is an input image with augmentation. The positive image is also same as the input image but different augmentation value. The negative image is a different image from the input image. In the first 5 epochs, the negative image is selected randomly from the dataset. After 5 epochs, choose the most difficult negative-positive image pairs in every step. To select a difficult pair, every epoch the model makes vectors of cards and picks a minimum distance from negative samples. Euclidean distance is used for measuring distance.


### Inference service

Use Python 3.11 and install `requirements.txt` in a virtual environment. Set
`MODEL_PATH` to the trained **distilled Detector state dictionary** (`best.pt`).
That checkpoint is not included in this repository; `weights/yolov8n_detector.pt`
is not an interchangeable checkpoint for this model.

```bash
MODEL_PATH=/absolute/path/to/best.pt chroma_mode=local chroma_path=./chroma \
  uvicorn server:app --host 0.0.0.0 --port 8000
```

The model and Chroma client load during application startup. Startup fails if they
cannot load. `/health` reports readiness after initialization; it is not a live
Chroma connectivity check. Synchronous inference and vector searches run in the
worker pool, keeping the event loop responsive. Each process allows one model
inference at a time and returns **503 + Retry-After: 1** when busy. More Uvicorn
workers load additional model copies, so size worker count for GPU memory.

Existing `/predict`, `/predict_embeds`, and `/search_embeds` response structures
are retained. Invalid images return 400, oversized images 413, invalid request
schemas 422, and embedding dimension/version mismatches 400. Internal failures
are logged server-side without exposing exception details in responses.

- `MAX_UPLOAD_BYTES`: decoded multipart file read limit, default 10 MiB. Apply a
  request-body limit at the ingress too, since multipart parsing precedes the handler.
- `MAX_IMAGE_PIXELS`: maximum width × height before RGB conversion, default 20 million.
- Embedding requests: 1–128 vectors, 1–4096 finite components per vector,
  consistent dimensions, nonzero vectors, and `top_k` between 1 and 100.
- `EMBED_DIM` and `EMBED_MODEL_VERSION` optionally enforce the deployed model contract.
- The public CORS configuration permits origins without cookie credentials.

### Regression tests

```bash
python3.11 -m venv .venv-test
.venv-test/bin/pip install -r requirements-test.txt
.venv-test/bin/python -m pytest -q
```

Run this suite on Linux (or WSL); snapshot imports and continuous evaluation use
POSIX file locks. GitHub Actions runs the same suite for pull requests and master.
These tests use real FastAPI request handling, image decoding, and snapshot files,
with fake model/Chroma adapters. They do not measure recognition accuracy or
validate GPU execution / a live Chroma deployment. `pytest.ini` keeps training
scripts such as `test_distillation.py` and the separate `tests/ml` suite out of
lightweight test discovery. With training dependencies installed, run the latter
explicitly using `python -m unittest discover -s tests/ml -v`.

### Edge Deployment (Client-side search)
- Run the AI service with local Chroma (embedded): set `chroma_mode=local` and mount a volume at `/chroma` (compose already set).
- Prepare a collection snapshot once from your central server:
  - `python scripts/export_chroma.py --out ./chroma_snapshot --host <central_host> --port 8000 --collection yugioh_256`
- Distribute `./chroma_snapshot` to each client server and import locally:
  - `python scripts/import_chroma.py --in ./chroma_snapshot --mode local --path /chroma --collection yugioh_256 --reset`
- After import, each client server performs vector search locally with no central dependency.

#### Packaging + Integrity
- Package snapshot to a single tar.gz with checksum:
  - `python scripts/pack_snapshot.py --src ./chroma_snapshot --out ./dist --name yugioh_256_YYYYMMDD`
  - Produces `./dist/yugioh_256_YYYYMMDD.tar.gz` and `.sha256`
- Optionally verify and extract elsewhere:
  - `python scripts/verify_and_extract.py --tgz ./dist/yugioh_256_YYYYMMDD.tar.gz --sha ./dist/yugioh_256_YYYYMMDD.sha256 --out ./verify --clean`

#### Auto-import on Container Start (local mode)
- Set environment variables (see docker-compose comments):
  - `AUTO_IMPORT=1`
  - `SNAPSHOT_TGZ=/chroma/snapshots/yugioh_256_YYYYMMDD.tar.gz`
  - `SNAPSHOT_SHA256=/chroma/snapshots/yugioh_256_YYYYMMDD.sha256`
  - optional: `IMPORT_ON_EMPTY=1` (default), `IMPORT_RESET=0`, `IMPORT_BATCH=1000`
- Mount the snapshot directory read-only to `/chroma/snapshots`.
- On startup, the container compares the actual archive checksum (if SHA is configured), validates data before resetting a collection, and records collection-specific `.snapshot_hash_<key>` files after success. A `.snapshot_pending_<key>` marker lets an interrupted import resume with idempotent upserts, including when `IMPORT_ON_EMPTY=1`.
- Configured missing snapshots/checksums, checksum mismatch, invalid data, and import failures stop startup. Set `AUTO_IMPORT=0` explicitly when no automatic import is wanted.
- Archives must match `pack_snapshot.py`: exactly three regular root files (`ids.json`, `metadatas.json`, `embeddings.npy`), at most 2 GiB unpacked. Links, extra paths, duplicate entries and special files are rejected.
- Validation catches empty/duplicate IDs, inconsistent lengths, malformed embedding dimensions and nonfinite values before a reset. A reset is not a database transaction: an operational failure after reset can leave a partial collection; restart with the same snapshot to resume. Manual imports also use upsert and can be rerun.
- Legacy `.snapshot_hash` markers are not reused for other collections. An existing nonempty collection remains untouched with `IMPORT_ON_EMPTY=1`; use `IMPORT_RESET=1` only when an intentional replacement is required.
