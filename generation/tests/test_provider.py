import base64
import json

import httpx
import pytest
from google import genai
from PIL import Image

from agile3d.models import ImageConfig
from agile3d.providers import generate_image


@pytest.mark.parametrize("api", ["generate_content", "interactions"])
def test_actual_google_sdk_serialization_and_image_decoding(api, tmp_path, monkeypatch):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (64, 64), "red").save(reference)
    encoded = base64.b64encode(reference.read_bytes()).decode()
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        if api == "interactions":
            data = dict(
                id="test",
                created="2026-10-08T00:00:00Z",
                updated="2026-10-08T00:00:00Z",
                status="completed",
                steps=[
                    dict(
                        type="model_output",
                        content=[
                            dict(type="image", mime_type="image/png", data=encoded)
                        ],
                    )
                ],
            )
        else:
            data = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": encoded,
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        return httpx.Response(200, json=data)

    original = genai.Client

    def client(**kwargs):
        kwargs["http_options"]["client_args"] = {
            "transport": httpx.MockTransport(handle),
            "trust_env": False,
        }
        kwargs["http_options"]["async_client_args"] = {"trust_env": False}
        return original(**kwargs)

    monkeypatch.setattr(genai, "Client", client)
    original_httpx = httpx.Client
    if api == "interactions":
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **kw: original_httpx(
                **kw, transport=httpx.MockTransport(handle), trust_env=False
            ),
        )
    monkeypatch.setenv("AGILE_TEST_KEY", "placeholder-not-a-real-key")
    output = tmp_path / "generated.png"
    metadata = generate_image(
        ImageConfig(api=api, model="test-image-model", api_key_env="AGILE_TEST_KEY"),
        [reference],
        "A target object",
        output,
    )
    assert Image.open(output).size == (64, 64)
    assert len(requests) == 1
    assert metadata["api"] == api
    assert metadata["model"] == "test-image-model"
    if api == "interactions":
        assert requests[0]["model"] == "test-image-model"


@pytest.mark.parametrize("api", ["generate_content", "interactions"])
def test_no_implicit_http_retry_on_server_failure(api, tmp_path, monkeypatch):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (64, 64)).save(reference)
    calls = []

    def fail(request):
        calls.append(1)
        return httpx.Response(
            503, json={"error": {"message": "test unavailable", "code": 503}}
        )

    original = genai.Client

    def client(**kwargs):
        kwargs["http_options"]["client_args"] = {
            "transport": httpx.MockTransport(fail),
            "trust_env": False,
        }
        kwargs["http_options"]["async_client_args"] = {"trust_env": False}
        return original(**kwargs)

    monkeypatch.setattr(genai, "Client", client)
    original_httpx = httpx.Client
    if api == "interactions":
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **kw: original_httpx(
                **kw, transport=httpx.MockTransport(fail), trust_env=False
            ),
        )
    monkeypatch.setenv("AGILE_TEST_KEY", "placeholder-not-a-real-key")
    with pytest.raises(Exception):
        generate_image(
            ImageConfig(api=api, model="test-image-model", api_key_env="AGILE_TEST_KEY"),
            [reference],
            "object",
            tmp_path / "out.png",
        )
    assert len(calls) == 1
