"""Ollama native-API provider: request shape, model routing, error paths."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cleanroom.config import Settings
from cleanroom.describe.vision import OllamaVisionClient, make_vision_client
from cleanroom.recreate.promptgen import OllamaTextClient, make_text_client
from cleanroom.utils import CleanroomError

REQUESTS = []


class FakeOllamaHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/api/chat", self.path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        REQUESTS.append(body)
        assert body["stream"] is False
        assert body["format"] == "json"
        # images must be raw base64 (Ollama native shape, no data: URL prefix)
        for m in body["messages"]:
            for img in m.get("images", []):
                assert not img.startswith("data:"), "native API takes raw base64"
                base64.b64decode(img)  # must decode cleanly
        content = json.dumps({"scene": "fake-ollama-saw-this"})
        resp = json.dumps({"message": {"role": "assistant", "content": content}, "done": True})
        data = resp.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def ollama_server():
    REQUESTS.clear()
    server = HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _frame(tmp_path):
    p = tmp_path / "f.jpg"
    p.write_bytes(b"\xff\xd8\xff fake jpeg bytes " * 100)
    from cleanroom.describe.extractor import FrameRef
    return FrameRef(timestamp=0.0, path=p)


def test_vision_routed_to_qwen(ollama_server, tmp_path):
    client = OllamaVisionClient(base_url=ollama_server)
    assert client._model == "qwen2.5vl:7b"  # default routing
    out = client.describe_scene(
        scene_id="s1", start_s=0.0, end_s=1.0,
        frames=[_frame(tmp_path), _frame(tmp_path)],
        system="sys", user="usr",
    )
    assert json.loads(out) == {"scene": "fake-ollama-saw-this"}
    body = REQUESTS[0]
    assert body["model"] == "qwen2.5vl:7b"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert len(body["messages"][1]["images"]) == 2


def test_vision_explicit_model(ollama_server, tmp_path):
    client = OllamaVisionClient(model="qwen2.5vl:3b", base_url=ollama_server)
    client.describe_scene(scene_id="s1", start_s=0, end_s=1,
                          frames=[_frame(tmp_path)], system="s", user="u")
    assert REQUESTS[0]["model"] == "qwen2.5vl:3b"


def test_text_client(ollama_server):
    client = OllamaTextClient(base_url=ollama_server)
    assert client._model == "qwen2.5:7b"
    out = client.complete("planner system", "plan this shot")
    assert json.loads(out) == {"scene": "fake-ollama-saw-this"}
    body = REQUESTS[0]
    assert body["model"] == "qwen2.5:7b"
    assert "images" not in body["messages"][1]


def test_factories_default_routing():
    s = Settings(vision_provider="ollama", text_provider="ollama")
    v, t = make_vision_client(s), make_text_client(s)
    assert isinstance(v, OllamaVisionClient) and v._model == "qwen2.5vl:7b"
    assert isinstance(t, OllamaTextClient) and t._model == "qwen2.5:7b"


def test_factories_explicit_model():
    s = Settings(vision_provider="ollama", vision_model="llava:7b",
                 text_provider="ollama", text_model="mistral:7b")
    assert make_vision_client(s)._model == "llava:7b"
    assert make_text_client(s)._model == "mistral:7b"


def test_unreachable_server_gives_helpful_error(tmp_path):
    client = OllamaVisionClient(base_url="http://127.0.0.1:1")
    with pytest.raises(CleanroomError, match="ollama pull qwen2.5vl:7b"):
        client.describe_scene(scene_id="s1", start_s=0, end_s=1,
                              frames=[_frame(tmp_path)], system="s", user="u")


def test_resolve_base_url_env(monkeypatch):
    from cleanroom.ollama import resolve_base_url
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/")
    assert resolve_base_url() == "http://gpu-box:11434"
    assert resolve_base_url("http://x:1/") == "http://x:1"
