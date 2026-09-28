import hashlib
import fcntl
import io
import json
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scripts import entrypoint, import_chroma
from scripts.snapshot_utils import extract_snapshot, load_snapshot, verify_checksum


def snapshot(tmp_path, ids=None, embeds=None):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    ids = ["one", "two"] if ids is None else ids
    (source / "ids.json").write_text(json.dumps(ids))
    (source / "metadatas.json").write_text(json.dumps([{"name": value} for value in ids]))
    np.save(source / "embeddings.npy", np.ones((len(ids), 2)) if embeds is None else embeds)
    archive = tmp_path / "snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in source.iterdir():
            tar.add(path, arcname=path.name)
    return source, archive


def setup_import(monkeypatch, tmp_path, archive):
    for key, value in {"AUTO_IMPORT": "1", "chroma_mode": "local", "chroma_path": str(tmp_path / "db"),
                       "SNAPSHOT_TGZ": str(archive), "SNAPSHOT_SHA256": "", "IMPORT_BATCH": "1",
                       "IMPORT_RESET": "0", "IMPORT_ON_EMPTY": "1", "chroma_collection": "cards"}.items():
        monkeypatch.setenv(key, value)


def test_valid_snapshot_roundtrip(tmp_path):
    source, archive = snapshot(tmp_path)
    checksum = tmp_path / "snapshot.sha256"
    checksum.write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "  snapshot.tar.gz")
    assert verify_checksum(archive, checksum)
    destination = tmp_path / "out"
    extract_snapshot(archive, destination)
    ids, metadata, embeds = load_snapshot(destination)
    assert ids == ["one", "two"]
    assert len(metadata) == 2
    assert embeds.shape == (2, 2)


@pytest.mark.parametrize("member_name,kind", [("../escape", tarfile.REGTYPE), ("/escape", tarfile.REGTYPE),
    ("ids.json", tarfile.SYMTYPE), ("ids.json", tarfile.LNKTYPE)])
def test_unsafe_archive_is_rejected_before_clean(tmp_path, member_name, kind):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(member_name)
        member.type = kind
        member.linkname = "../outside"
        member.size = 0
        tar.addfile(member, io.BytesIO())
        for name in {"ids.json", "metadatas.json", "embeddings.npy"} - {member_name}:
            tar.addfile(tarfile.TarInfo(name), io.BytesIO())
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "keep.txt"
    existing.write_text("keep")
    with pytest.raises(ValueError):
        extract_snapshot(archive, out, clean=True)
    assert existing.read_text() == "keep"
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("ids,embeds", [([], np.ones((0, 2))), (["a", "a"], np.ones((2, 2))),
    (["a"], np.ones((2, 2))), (["a"], np.array([1.])), (["a"], np.array([[np.nan]]))])
def test_invalid_snapshot_data(tmp_path, ids, embeds):
    source, _ = snapshot(tmp_path, ids, embeds)
    with pytest.raises(ValueError):
        load_snapshot(source)


def test_checksum_is_checked_even_when_marker_claims_imported(tmp_path, monkeypatch):
    _, archive = snapshot(tmp_path)
    setup_import(monkeypatch, tmp_path, archive)
    checksum = tmp_path / "bad.sha256"
    checksum.write_text("0" * 64)
    monkeypatch.setenv("SNAPSHOT_SHA256", str(checksum))
    db = tmp_path / "db"
    db.mkdir()
    key = hashlib.sha256(b"cards").hexdigest()[:16]
    (db / f".snapshot_hash_{key}").write_text("0" * 64)
    client_factory = Mock()
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=client_factory))
    with pytest.raises(ValueError, match="checksum"):
        entrypoint.main()
    client_factory.assert_not_called()


def test_manual_reset_validates_before_opening_database(tmp_path, monkeypatch):
    source, _ = snapshot(tmp_path, ["one"], np.ones((2, 2)))
    monkeypatch.setattr(sys, "argv", ["import_chroma.py", "--in", str(source), "--reset"])
    client_factory = Mock()
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=client_factory))
    with pytest.raises(ValueError):
        import_chroma.main()
    client_factory.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_manual_import_preserves_index_lock_and_revision(tmp_path, monkeypatch, fail):
    source, _ = snapshot(tmp_path)
    state = tmp_path / "state"
    monkeypatch.setattr(sys, "argv", ["import_chroma.py", "--in", str(source),
        "--state-dir", str(state), "--path", str(tmp_path / "db"), "--batch", "1"])

    def upsert(**kwargs):
        assert json.loads((state / "external-index.json").read_text())["pending"] is True
        with (state / "cycle.lock").open("w") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if fail:
            raise RuntimeError("interrupted import")

    collection = SimpleNamespace(upsert=Mock(side_effect=upsert))
    database = SimpleNamespace(get_or_create_collection=lambda *a, **k: collection)
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **_: database))
    if fail:
        with pytest.raises(RuntimeError, match="interrupted import"):
            import_chroma.main()
    else:
        import_chroma.main()
        assert collection.upsert.call_count == 2
    marker = json.loads((state / "external-index.json").read_text())
    assert marker["pending"] is fail
    assert marker["revision"]
    with (state / "cycle.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("value", [None, [1, 2], {"nested": "value"}, float("nan")])
def test_unsupported_metadata_is_rejected(tmp_path, value):
    source, _ = snapshot(tmp_path, ["one"])
    (source / "metadatas.json").write_text(json.dumps([{"name": value}]))
    with pytest.raises(ValueError, match="metadata"):
        load_snapshot(source)


def test_auto_reset_preserves_collection_on_invalid_snapshot(tmp_path, monkeypatch):
    _, archive = snapshot(tmp_path, ["one"], np.ones((2, 2)))
    setup_import(monkeypatch, tmp_path, archive)
    monkeypatch.setenv("IMPORT_RESET", "1")
    database = Mock()
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **_: database))
    with pytest.raises(ValueError):
        entrypoint.main()
    database.delete_collection.assert_not_called()


@pytest.mark.parametrize("prior_completed_marker", [False, True])
def test_interrupted_import_resumes_without_skipping_partial_data(tmp_path, monkeypatch, prior_completed_marker):
    _, archive = snapshot(tmp_path)
    setup_import(monkeypatch, tmp_path, archive)
    items = {}
    calls = 0

    def upsert(ids, metadatas, embeddings):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("temporary failure")
        items.update(zip(ids, embeddings))

    collection = SimpleNamespace(count=lambda: len(items), upsert=upsert)
    database = SimpleNamespace(get_or_create_collection=lambda *a, **k: collection)
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **_: database))
    with pytest.raises(RuntimeError, match="temporary"):
        entrypoint.main()
    assert len(items) == 1
    assert not list((tmp_path / "db").glob(".snapshot_hash_*"))
    if prior_completed_marker:
        # A reset of an already imported archive can fail after deleting old data.
        key = hashlib.sha256(b"cards").hexdigest()[:16]
        (tmp_path / "db" / f".snapshot_hash_{key}").write_text(verify_checksum(archive))
    entrypoint.main()
    assert set(items) == {"one", "two"}
    assert not list((tmp_path / "db").glob(".snapshot_pending_*"))
    assert len(list((tmp_path / "db").glob(".snapshot_hash_*"))) == 1
