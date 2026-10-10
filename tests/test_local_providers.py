"""Local (offline) providers: request shape, factory mapping, JSON tolerance."""
import sys
import types
from unittest import mock

import pytest

from cleanroom.config import Settings
from cleanroom.describe.vision import (
    LocalVisionClient,
    OpenAIVisionClient,
    make_vision_client,
)
from cleanroom.recreate.promptgen import (
    LocalTextClient,
    OpenAITextClient,
    make_text_client,
)
from cleanroom.utils import CleanroomError, extract_json


@pytest.fixture()
def fake_openai(monkeypatch):
    """Stub the `openai` package: records create() kwargs, returns canned JSON."""
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls["create_kwargs"] = kwargs
            msg = mock.Mock()
            msg.content = '```json\n{"prompt": "a cat", "negative_prompt": "blur"}\n```'
            return mock.Mock(choices=[mock.Mock(message=msg)])

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs
            self.chat = FakeChat()

    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return calls


def _vision_call(client, fake_openai, tmp_path):
    img = tmp_path / "f.jpg"
    img.write_bytes(b"\xff\xd8\xff fake jpeg")
    from cleanroom.describe.extractor import FrameRef
    client.describe_scene(
        scene_id="s1", start_s=0.0, end_s=1.0,
        frames=[FrameRef(timestamp=0.0, path=img)],
        system="sys", user="usr",
    )
    return fake_openai["create_kwargs"]


def test_openai_vision_uses_json_mode(fake_openai, tmp_path):
    kwargs = _vision_call(OpenAIVisionClient(model="m", api_key="k"), fake_openai, tmp_path)
    assert kwargs["response_format"] == {"type": "json_object"}
    # image sent as data URL, OpenAI format
    content = kwargs["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "usr"}
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_local_vision_skips_json_mode_and_key(fake_openai, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LocalVisionClient(model="qwen2.5vl:7b", base_url="http://localhost:11434/v1")
    kwargs = _vision_call(client, fake_openai, tmp_path)
    assert "response_format" not in kwargs
    assert fake_openai["init_kwargs"]["base_url"] == "http://localhost:11434/v1"
    assert fake_openai["init_kwargs"]["api_key"] == "sk-local"


def test_local_text_client_no_json_mode(fake_openai, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LocalTextClient(model="qwen2.5:7b")
    client.complete("sys", "usr")
    assert "response_format" not in fake_openai["create_kwargs"]


def test_openai_text_client_uses_json_mode(fake_openai):
    client = OpenAITextClient(model="gpt-4o", api_key="k")
    client.complete("sys", "usr")
    assert fake_openai["create_kwargs"]["response_format"] == {"type": "json_object"}


def test_factories_map_local(fake_openai):
    s = Settings(vision_provider="local", text_provider="local")
    assert isinstance(make_vision_client(s), LocalVisionClient)
    assert isinstance(make_text_client(s), LocalTextClient)
    assert make_vision_client(s).name == "local"


def test_extract_json_strips_fences():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('  {"a": 1}  ') == '{"a": 1}'
    assert extract_json('note {"a": 1} trailing') == '{"a": 1}'


def test_extract_json_no_object_raises():
    with pytest.raises(CleanroomError):
        extract_json("no json here at all")
