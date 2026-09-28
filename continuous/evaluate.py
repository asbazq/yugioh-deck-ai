"""Evaluate labeled deck screenshots through the production prediction path."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path


def load_samples(manifest):
    manifest = Path(manifest)
    rows = json.loads(manifest.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation manifest must contain labeled screenshots")
    samples, seen = [], set()
    digest = hashlib.sha256()
    for row in rows:
        path = (manifest.parent / row["image"]).resolve()
        ids = row["card_ids"]
        if not isinstance(ids, list) or not ids or any(
            isinstance(x, bool) or not isinstance(x, (str, int)) or not str(x)
            for x in ids
        ):
            raise ValueError("card_ids must be a nonempty list of card IDs")
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if content_hash in seen:
            raise ValueError("Duplicate evaluation image")
        seen.add(content_hash)
        ids = [str(x) for x in ids]
        digest.update(json.dumps([content_hash, sorted(ids)]).encode())
        samples.append((path, ids))
    return samples, digest.hexdigest()


def score_decks(expected, predicted):
    """Micro F1 counts missed, extra and duplicate cards, independent of box order."""
    if not expected or len(expected) != len(predicted):
        raise ValueError("Expected and predicted decks must have equal nonzero length")
    matched = truth_count = pred_count = exact = 0
    for truth, pred in zip(expected, predicted):
        truth, pred = Counter(map(str, truth)), Counter(map(str, pred))
        matched += sum((truth & pred).values())
        truth_count += sum(truth.values())
        pred_count += sum(pred.values())
        exact += truth == pred
    if truth_count == 0:
        raise ValueError("Evaluation requires labeled cards")
    return {
        "samples": len(expected), "cards": truth_count,
        "f1": 2 * matched / (truth_count + pred_count),
        "precision": matched / pred_count if pred_count else 0.0,
        "recall": matched / truth_count, "exact_deck_accuracy": exact / len(expected),
    }


def validate_report(report):
    for key in ("f1", "precision", "recall", "exact_deck_accuracy"):
        if not math.isfinite(report[key]) or not 0 <= report[key] <= 1:
            raise ValueError(f"Invalid evaluation metric: {key}")
    if report["samples"] <= 0 or report["cards"] <= 0:
        raise ValueError("Empty evaluation")
    return report


async def evaluate(samples):
    # Import after MODEL_PATH is set; use exactly the deployed preprocessing/search.
    from fastapi import Request, UploadFile
    from starlette.concurrency import run_in_threadpool
    from starlette.datastructures import Headers
    from server import TorchRuntime, create_app

    application = create_app(TorchRuntime)
    predicted = []
    async with application.router.lifespan_context(application):
        count = await run_in_threadpool(application.state.runtime.chroma.collection.count)
        if count == 0:
            raise ValueError("Evaluation requires a populated production vector collection")
        predict = next(route.endpoint for route in application.routes if route.path == "/predict")
        request = Request({"type": "http", "app": application})
        for path, _ in samples:
            content_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                            ".png": "image/png", ".webp": "image/webp"}[path.suffix.lower()]
            with path.open("rb") as stream:
                result = await run_in_threadpool(predict, request, UploadFile(
                    file=stream, filename=path.name,
                    headers=Headers({"content-type": content_type}),
                ))
            predicted.append([d.best.id for d in result.detections])
    return predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["MODEL_PATH"] = str(Path(args.checkpoint).resolve())
    samples, fingerprint = load_samples(args.manifest)
    predicted = asyncio.run(evaluate(samples))
    report = validate_report(score_decks([ids for _, ids in samples], predicted))
    report["dataset_sha256"] = fingerprint
    report["checkpoint_sha256"] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
