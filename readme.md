# Yu-Gi-Oh card finder


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


### Demo
```
python test.py path/to/image.png
```

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
- On startup, the container will verify (if SHA provided), import if needed, and mark the imported hash in `/chroma/.snapshot_hash` to avoid repeated imports.
