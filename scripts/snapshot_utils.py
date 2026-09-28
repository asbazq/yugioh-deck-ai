"""Validation shared by manual and automatic snapshot imports."""
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path

SNAPSHOT_FILES = {"ids.json", "metadatas.json", "embeddings.npy"}
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024 * 1024


def verify_checksum(archive: Path, checksum: Path | None = None) -> str:
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if checksum is not None:
        fields = checksum.read_text(encoding="utf-8").split()
        if not fields or fields[0].lower() != actual:
            raise ValueError("Snapshot checksum mismatch")
    return actual


def extract_snapshot(archive: Path, destination: Path, clean: bool = False):
    """Accept only the three regular files produced by pack_snapshot.py."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if set(names) != SNAPSHOT_FILES or len(names) != len(SNAPSHOT_FILES):
            raise ValueError("Snapshot must contain exactly ids.json, metadatas.json and embeddings.npy")
        if any(not member.isfile() for member in members):
            raise ValueError("Snapshot links and special files are not allowed")
        if sum(member.size for member in members) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Snapshot exceeds the extraction size limit")
        root = destination.resolve()
        if root == Path(root.anchor) or archive.resolve().is_relative_to(root):
            raise ValueError("Unsafe snapshot destination")
        # Validation precedes --clean, so invalid archives cannot erase existing output.
        if clean and destination.exists():
            if destination.is_symlink():
                raise ValueError("Cannot clean a symlink destination")
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = destination / member.name
            if target.is_symlink():
                raise ValueError("Snapshot output must not be a symlink")
        for member in members:
            with tar.extractfile(member) as source, (destination / member.name).open("wb") as target:
                shutil.copyfileobj(source, target)


def load_snapshot(directory: Path):
    import numpy as np

    ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadatas.json").read_text(encoding="utf-8"))
    embeds = np.load(directory / "embeddings.npy", allow_pickle=False)
    if not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Snapshot IDs must be a nonempty list of nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Snapshot IDs must be unique")
    if not isinstance(metadata, list) or len(metadata) != len(ids):
        raise ValueError("Snapshot parts have different lengths")
    if any(not isinstance(item, dict) or not item for item in metadata):
        raise ValueError("Snapshot metadata must contain nonempty objects")
    for item in metadata:
        for key, value in item.items():
            if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
                raise ValueError("Snapshot metadata values must be scalar strings, numbers or booleans")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Snapshot metadata numbers must be finite")
    if embeds.ndim != 2 or embeds.shape[0] != len(ids) or embeds.shape[1] == 0:
        raise ValueError("Snapshot embeddings must have shape (number of IDs, dimension)")
    if not np.issubdtype(embeds.dtype, np.number) or np.iscomplexobj(embeds) or not np.isfinite(embeds).all():
        raise ValueError("Snapshot embeddings must be finite real numbers")
    return ids, metadata, embeds
