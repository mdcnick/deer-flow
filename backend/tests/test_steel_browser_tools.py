"""Unit tests for the Steel Browser capture tool."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from deerflow.community.steel_browser import tools as steel_tools


@pytest.fixture
def mock_config():
    with patch("deerflow.community.steel_browser.tools.get_app_config") as mock:
        tool_config = MagicMock()
        tool_config.model_extra = {
            "base_url": "https://steel.example.test",
            "api_key": "test-key",
            "timeout": 10,
            "max_capture_bytes": 1024 * 1024,
        }
        mock.return_value.get_tool_config.return_value = tool_config
        yield mock


@pytest.fixture
def runtime(tmp_path):
    outputs = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    return SimpleNamespace(
        state={
            "sandbox": {"sandbox_id": "local"},
            "thread_data": {
                "workspace_path": str(tmp_path / "threads" / "thread-1" / "user-data" / "workspace"),
                "uploads_path": str(tmp_path / "threads" / "thread-1" / "user-data" / "uploads"),
                "outputs_path": str(outputs),
            },
        },
        context={"thread_id": "thread-1"},
        config={"configurable": {"thread_id": "thread-1"}},
    )


def _public_dns(*_args, **_kwargs):
    return [(None, None, None, None, ("93.184.216.34", 0))]


def _private_dns(*_args, **_kwargs):
    return [(None, None, None, None, ("127.0.0.1", 0))]


class TestCaptureUrlValidation:
    def test_rejects_non_http_schemes(self):
        with pytest.raises(steel_tools.CaptureUrlError):
            steel_tools._validate_capture_url("file:///etc/passwd")

    def test_rejects_embedded_credentials(self):
        with pytest.raises(steel_tools.CaptureUrlError):
            steel_tools._validate_capture_url("https://user:pass@example.com")

    def test_rejects_private_resolved_addresses(self):
        with patch("deerflow.community.steel_browser.tools.socket.getaddrinfo", side_effect=_private_dns):
            with pytest.raises(steel_tools.CaptureUrlError):
                steel_tools._validate_capture_url("https://example.com")

    def test_allows_public_http_urls(self):
        with patch("deerflow.community.steel_browser.tools.socket.getaddrinfo", side_effect=_public_dns):
            assert steel_tools._validate_capture_url("https://example.com/path") == "https://example.com/path"


@pytest.mark.asyncio
async def test_capture_writes_screenshot_artifact(mock_config, runtime):
    async def fake_validate(url, *, timeout):
        return url

    async def fake_post_binary(_client, endpoint, payload, *, max_bytes, artifact):
        assert endpoint == "/v1/screenshot"
        assert payload == {"url": "https://example.com", "delay": 1000, "fullPage": True}
        assert artifact == "screenshot"
        assert max_bytes == 1024 * 1024
        return b"jpeg-bytes"

    with patch("deerflow.community.steel_browser.tools._validate_redirect_chain", side_effect=fake_validate), patch("deerflow.community.steel_browser.tools._post_binary", side_effect=fake_post_binary):
        result = await steel_tools.steel_capture_tool.coroutine(runtime, "https://example.com")

    parsed = json.loads(result)
    assert parsed["url"] == "https://example.com"
    assert parsed["artifacts"]["screenshot"] == "/mnt/user-data/outputs/example.com.jpg"
    assert (tmp_path := runtime.state["thread_data"]["outputs_path"])
    assert (steel_tools.Path(tmp_path) / "example.com.jpg").read_bytes() == b"jpeg-bytes"


@pytest.mark.asyncio
async def test_capture_uses_session_endpoints_when_recording(mock_config, runtime):
    async def fake_validate(url, *, timeout):
        return url

    async def fake_post_json(_client, endpoint, payload):
        assert endpoint == "/v1/sessions"
        assert payload["blockAds"] is True
        return {"id": "session-1", "sessionViewerUrl": "https://steel.example.test/ui/session-1"}

    async def fake_post_binary(_client, endpoint, payload, *, max_bytes, artifact):
        assert endpoint == "/v1/sessions/screenshot"
        return b"jpeg-bytes"

    released = []

    async def fake_release(_client, session_id):
        released.append(session_id)

    with (
        patch("deerflow.community.steel_browser.tools._validate_redirect_chain", side_effect=fake_validate),
        patch("deerflow.community.steel_browser.tools._post_json", side_effect=fake_post_json),
        patch("deerflow.community.steel_browser.tools._post_binary", side_effect=fake_post_binary),
        patch("deerflow.community.steel_browser.tools._release_session", side_effect=fake_release),
    ):
        result = await steel_tools.steel_capture_tool.coroutine(runtime, "https://example.com", record_session=True)

    parsed = json.loads(result)
    assert parsed["steel_session_id"] == "session-1"
    assert parsed["steel_session_viewer_url"] == "https://steel.example.test/ui/session-1"
    assert released == ["session-1"]


@pytest.mark.asyncio
async def test_missing_base_url_returns_error(runtime):
    with patch("deerflow.community.steel_browser.tools.get_app_config") as mock:
        tool_config = MagicMock()
        tool_config.model_extra = {}
        mock.return_value.get_tool_config.return_value = tool_config

        result = await steel_tools.steel_capture_tool.coroutine(runtime, "https://example.com")

    parsed = json.loads(result)
    assert "STEEL_BASE_URL" in parsed["error"]
