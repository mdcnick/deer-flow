"""Steel Browser capture tool for real-browser web evidence."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

from deerflow.config import get_app_config
from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.sandbox.tools import ensure_thread_directories_exist, get_thread_data
from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_DELAY_MS = 1000
_DEFAULT_MAX_CAPTURE_BYTES = 15 * 1024 * 1024
_DEFAULT_DIMENSIONS = {"width": 1280, "height": 900}
_STEEL_API_KEY_ENV = "STEEL_API_KEY"
_STEEL_BASE_URL_ENV = "STEEL_BASE_URL"
_TOOL_NAME = "steel_capture"
_SAFE_FILENAME_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")
_BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain"}
_BLOCKED_METADATA_HOSTS = {"metadata.google.internal"}


class CaptureUrlError(ValueError):
    """Raised when a capture URL is not safe to load server-side."""


def _json_error(message: str, *, url: str | None = None) -> str:
    payload: dict[str, Any] = {"error": message}
    if url is not None:
        payload["url"] = url
    return json.dumps(payload, ensure_ascii=False)


def _config_extra() -> dict[str, Any]:
    config = get_app_config().get_tool_config(_TOOL_NAME)
    if config is None:
        return {}
    return dict(config.model_extra or {})


def _string_from_config(extra: dict[str, Any], key: str, env_key: str | None = None) -> str | None:
    value = extra.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if env_key:
        env_value = os.getenv(env_key)
        if env_value and env_value.strip():
            return env_value.strip()
    return None


def _int_from_config(extra: dict[str, Any], key: str, default: int, *, minimum: int, maximum: int) -> int:
    value = extra.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _float_from_config(extra: dict[str, Any], key: str, default: float, *, minimum: float, maximum: float) -> float:
    value = extra.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _bool_from_config(extra: dict[str, Any], key: str, default: bool) -> bool:
    value = extra.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    stripped = base_url.rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return stripped


def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"steel-api-key": api_key}


def _is_blocked_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


def _validate_capture_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise CaptureUrlError("Only http and https URLs can be captured")
    if not parsed.netloc or not parsed.hostname:
        raise CaptureUrlError("Capture URL must include a hostname")
    if parsed.username or parsed.password:
        raise CaptureUrlError("Capture URLs must not contain embedded credentials")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in _BLOCKED_HOSTNAMES or hostname in _BLOCKED_METADATA_HOSTS:
        raise CaptureUrlError("Capture URL hostname is not allowed")

    try:
        addr_infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise CaptureUrlError("Capture URL hostname could not be resolved") from exc

    addresses = {info[4][0] for info in addr_infos}
    if not addresses:
        raise CaptureUrlError("Capture URL hostname did not resolve to an address")
    if any(_is_blocked_ip(address) for address in addresses):
        raise CaptureUrlError("Capture URL resolves to a private or reserved network address")

    return url


async def _validate_redirect_chain(url: str, *, timeout: httpx.Timeout) -> str:
    _validate_capture_url(url)
    try:
        async with httpx.AsyncClient(timeout=timeout, max_redirects=5) as client:
            async with client.stream("GET", url, headers={"Range": "bytes=0-0"}, follow_redirects=True) as response:
                for hop in response.history:
                    _validate_capture_url(str(hop.url))
                final_url = str(response.url)
                _validate_capture_url(final_url)
                return final_url
    except CaptureUrlError:
        raise
    except httpx.TooManyRedirects as exc:
        raise CaptureUrlError("Capture URL has too many redirects") from exc
    except httpx.HTTPError as exc:
        raise CaptureUrlError(f"Capture URL could not be preflighted: {exc.__class__.__name__}") from exc


def _safe_capture_stem(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or "capture"
    path = parsed.path.strip("/").replace("/", "-")
    stem = f"{host}-{path}" if path else host
    stem = _SAFE_FILENAME_CHARS.sub("-", stem).strip("-._").lower()
    return stem[:80] or "capture"


def _unique_output_path(outputs_dir: Path, stem: str, suffix: str) -> Path:
    candidate = outputs_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        candidate = outputs_dir / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a unique output filename")


def _get_outputs_dir(runtime: Runtime) -> Path:
    ensure_thread_directories_exist(runtime)
    thread_data = get_thread_data(runtime)
    if thread_data is None or not thread_data.get("outputs_path"):
        raise RuntimeError("Thread outputs directory is not available")
    outputs_dir = Path(thread_data["outputs_path"]).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def _virtual_output_path(path: Path) -> str:
    return f"{VIRTUAL_PATH_PREFIX}/outputs/{path.name}"


def _raise_for_oversize(content: bytes, max_bytes: int, artifact: str) -> None:
    if len(content) > max_bytes:
        raise RuntimeError(f"Steel {artifact} response exceeded configured maximum size")


async def _post_binary(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    *,
    max_bytes: int,
    artifact: str,
) -> bytes:
    response = await client.post(endpoint, json=payload)
    response.raise_for_status()
    content = response.content
    _raise_for_oversize(content, max_bytes, artifact)
    return content


async def _post_json(client: httpx.AsyncClient, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = await client.post(endpoint, json=payload)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Steel returned a non-object JSON response")
    return data


async def _release_session(client: httpx.AsyncClient, session_id: str) -> None:
    try:
        await client.post(f"/v1/sessions/{session_id}/release")
    except httpx.HTTPError as exc:
        logger.warning("Failed to release Steel session %s: %s", session_id, exc.__class__.__name__)


@tool("steel_capture", parse_docstring=True)
async def steel_capture_tool(
    runtime: Runtime,
    url: str,
    screenshot: bool = True,
    pdf: bool = False,
    full_page: bool = True,
    delay_ms: int = _DEFAULT_DELAY_MS,
    record_session: bool = False,
) -> str:
    """Capture a website with a real Steel Browser instance.

    Use this after web_search or web_fetch returns an exact URL when the user wants visual evidence,
    screenshots, PDFs, or a Steel session viewer link. Only capture exact http/https URLs from the
    user or prior web results; do not guess URLs.

    Args:
        url: Exact http or https URL to capture.
        screenshot: Save a screenshot artifact in /mnt/user-data/outputs.
        pdf: Save a PDF artifact in /mnt/user-data/outputs.
        full_page: Capture the full page when saving a screenshot.
        delay_ms: Milliseconds Steel should wait after page load before capture.
        record_session: Use Steel's session endpoints and return the session viewer URL when available.
    """
    extra = _config_extra()
    base_url = _normalize_base_url(_string_from_config(extra, "base_url", _STEEL_BASE_URL_ENV))
    if base_url is None:
        return _json_error("STEEL_BASE_URL or steel_capture.base_url must be configured", url=url)

    if not screenshot and not pdf:
        return _json_error("At least one of screenshot or pdf must be true", url=url)

    timeout_seconds = _float_from_config(extra, "timeout", _DEFAULT_TIMEOUT_SECONDS, minimum=1.0, maximum=120.0)
    max_bytes = _int_from_config(extra, "max_capture_bytes", _DEFAULT_MAX_CAPTURE_BYTES, minimum=1024, maximum=100 * 1024 * 1024)
    configured_delay_ms = _int_from_config(extra, "delay_ms", _DEFAULT_DELAY_MS, minimum=0, maximum=30000)
    if delay_ms == _DEFAULT_DELAY_MS:
        delay_ms = configured_delay_ms
    else:
        delay_ms = max(0, min(30000, int(delay_ms)))

    block_ads = _bool_from_config(extra, "block_ads", True)
    release_session = _bool_from_config(extra, "release_session", True)
    api_key = _string_from_config(extra, "api_key", _STEEL_API_KEY_ENV)
    dimensions = extra.get("dimensions") if isinstance(extra.get("dimensions"), dict) else _DEFAULT_DIMENSIONS

    try:
        outputs_dir = _get_outputs_dir(runtime)
    except Exception as exc:
        return _json_error(str(exc), url=url)

    timeout = httpx.Timeout(timeout_seconds)
    try:
        final_url = await _validate_redirect_chain(url, timeout=timeout)
    except CaptureUrlError as exc:
        return _json_error(str(exc), url=url)

    async with httpx.AsyncClient(base_url=base_url, headers=_headers(api_key), timeout=timeout, max_redirects=5) as client:
        payload = {"url": final_url, "delay": delay_ms}
        session_id: str | None = None
        session_viewer_url: str | None = None
        endpoint_prefix = "/v1"

        try:
            if record_session:
                session_payload: dict[str, Any] = {"blockAds": block_ads, "dimensions": dimensions}
                session = await _post_json(client, "/v1/sessions", session_payload)
                session_id = session.get("id") if isinstance(session.get("id"), str) else None
                session_viewer_url = session.get("sessionViewerUrl") if isinstance(session.get("sessionViewerUrl"), str) else None
                endpoint_prefix = "/v1/sessions"

            result: dict[str, Any] = {
                "url": url,
                "final_url": final_url,
                "artifacts": {},
            }
            if session_id:
                result["steel_session_id"] = session_id
            if session_viewer_url:
                result["steel_session_viewer_url"] = session_viewer_url

            stem = _safe_capture_stem(final_url)
            if screenshot:
                image = await _post_binary(
                    client,
                    f"{endpoint_prefix}/screenshot",
                    {**payload, "fullPage": full_page},
                    max_bytes=max_bytes,
                    artifact="screenshot",
                )
                image_path = _unique_output_path(outputs_dir, stem, ".jpg")
                image_path.write_bytes(image)
                result["artifacts"]["screenshot"] = _virtual_output_path(image_path)

            if pdf:
                pdf_bytes = await _post_binary(
                    client,
                    f"{endpoint_prefix}/pdf",
                    payload,
                    max_bytes=max_bytes,
                    artifact="pdf",
                )
                pdf_path = _unique_output_path(outputs_dir, stem, ".pdf")
                pdf_path.write_bytes(pdf_bytes)
                result["artifacts"]["pdf"] = _virtual_output_path(pdf_path)

            return json.dumps(result, indent=2, ensure_ascii=False)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            return _json_error(f"Steel request failed with HTTP {status}", url=url)
        except httpx.HTTPError as exc:
            return _json_error(f"Steel request failed: {exc.__class__.__name__}", url=url)
        except Exception as exc:
            return _json_error(str(exc), url=url)
        finally:
            if record_session and release_session and session_id:
                await _release_session(client, session_id)
