import asyncio
import io
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import server
from continuous.evaluate import evaluate


def png():
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), (255, 0, 0)).save(stream, format="PNG")
    return stream.getvalue()


class Runtime:
    def __init__(self):
        self.searches = []

    def infer(self, image):
        assert image.shape == (6, 8, 3)
        assert image[0, 0].tolist() == [255, 0, 0]
        return [SimpleNamespace(bboxes=np.array([[1, 2, 3, 4]]), embeds=np.array([[1., 2.]]))]

    def search(self, embed, top_k):
        self.searches.append((embed, top_k))
        return [{"id": 123, "name": "Card"}]


@pytest.mark.parametrize("count", [0, 1])
def test_continuous_evaluation_uses_serving_lifespan(tmp_path, monkeypatch, count):
    runtime = Runtime()
    runtime.chroma = SimpleNamespace(collection=SimpleNamespace(count=lambda: count))
    monkeypatch.setattr(server, "TorchRuntime", lambda: runtime)
    applications = []
    create_app = server.create_app

    def capture_app(factory):
        application = create_app(factory)
        applications.append(application)
        return application

    monkeypatch.setattr(server, "create_app", capture_app)
    path = tmp_path / "card.png"
    path.write_bytes(png())
    if count:
        assert asyncio.run(evaluate([(path, [123])])) == [[123]]
        assert runtime.searches == [([1., 2.], server.TOP_K)]
    else:
        with pytest.raises(ValueError, match="populated"):
            asyncio.run(evaluate([(path, [123])]))
        assert runtime.searches == []
    assert applications[0].state.runtime is None


@pytest.fixture
def client():
    with TestClient(server.create_app(Runtime)) as client:
        yield client


@pytest.mark.parametrize("route", ["/predict", "/predict_embeds"])
def test_image_response_contract(client, route):
    response = client.post(route, files={"file": ("card.png", png(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["elapsed"] >= 0
    if route == "/predict":
        assert result["detections"][0]["best"] == {"id": 123, "name": "Card"}
    else:
        assert result["detections"][0] == {"bbox": [1, 2, 3, 4], "embed": [1, 2]}


@pytest.mark.parametrize("route", ["/predict", "/predict_embeds"])
@pytest.mark.parametrize("raw,content_type", [(b"", "image/png"), (b"broken", "image/png"), (b"broken", "text/plain")])
def test_bad_upload_returns_client_error_and_releases_slot(client, route, raw, content_type):
    assert client.post(route, files={"file": ("bad", raw, content_type)}).status_code == 400
    assert client.post(route, files={"file": ("card.png", png(), "image/png")}).status_code == 200


@pytest.mark.parametrize("setting,value", [("MAX_UPLOAD_BYTES", 10), ("MAX_IMAGE_PIXELS", 10)])
def test_upload_limits(client, monkeypatch, setting, value):
    monkeypatch.setattr(server, setting, value)
    assert client.post("/predict", files={"file": ("card.png", png(), "image/png")}).status_code == 413


@pytest.mark.parametrize("payload", [
    {"embeds": []}, {"embeds": [[]]}, {"embeds": [[1.0]], "top_k": 0},
    {"embeds": [[1.0]], "top_k": 101}, {"embeds": [["NaN"]]},
    {"embeds": [[1.0]] * 129}, {"embeds": [[1.0] * 4097]},
])
def test_invalid_embedding_payloads(client, payload):
    assert client.post("/search_embeds", json=payload).status_code == 422


def test_raw_nonfinite_json_returns_validation_error(client):
    assert client.post("/search_embeds", content='{"embeds":[[NaN]]}', headers={"Content-Type": "application/json"}).status_code == 422


@pytest.mark.parametrize("embeds", [[[0., 0.]], [[1., 2.], [3.]]])
def test_zero_or_inconsistent_vectors(client, embeds):
    assert client.post("/search_embeds", json={"embeds": embeds}).status_code == 400


def test_version_and_dimension(client, monkeypatch):
    monkeypatch.setattr(server, "EMBED_DIM", 2)
    monkeypatch.setattr(server, "EMBED_MODEL_VERSION", "v1")
    assert client.post("/search_embeds", json={"embeds": [[1., 2.]]}).status_code == 400
    assert client.post("/search_embeds", json={"embeds": [[1.]], "model_version": "v1"}).status_code == 400
    response = client.post("/search_embeds", json={"embeds": [[1., 2.]], "model_version": "v1", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["results"][0]["best"]["id"] == 123
    assert client.app.state.runtime.searches == [([1., 2.], 2)]


def test_empty_search_results_preserve_response_shape(client):
    client.app.state.runtime.search = lambda *_: []
    response = client.post("/search_embeds", json={"embeds": [[1.]]})
    assert response.json()["results"] == [{"best": {"id": None, "name": None}, "topk": []}]


def test_health_is_not_ready_without_lifespan():
    client = TestClient(server.create_app(Runtime))
    assert client.get("/health").status_code == 503


def test_internal_error_is_logged_not_exposed(client):
    def fail(*_):
        raise RuntimeError("secret-database-address")
    client.app.state.runtime.search = fail
    response = client.post("/search_embeds", json={"embeds": [[1.]]})
    assert response.status_code == 500
    assert "secret" not in response.text


def test_inference_does_not_block_health_and_overload_is_rejected():
    started, release = threading.Event(), threading.Event()

    class SlowRuntime(Runtime):
        def infer(self, image):
            started.set()
            if not release.wait(3):
                raise RuntimeError("Test inference release timed out")
            return super().infer(image)

    async def scenario():
        application = server.create_app(SlowRuntime)
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
                pending = asyncio.create_task(client.post("/predict", files={"file": ("card.png", png(), "image/png")}))
                try:
                    assert await asyncio.to_thread(started.wait, 1)
                    health = await asyncio.wait_for(client.get("/health"), 1)
                    assert health.status_code == 200
                    overloaded = await asyncio.wait_for(client.post("/predict_embeds", files={"file": ("card.png", png(), "image/png")}), 1)
                    assert overloaded.status_code == 503
                    assert overloaded.headers["retry-after"] == "1"
                finally:
                    release.set()
                    result = await pending
                assert result.status_code == 200

    asyncio.run(scenario())
