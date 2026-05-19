"""Web search tool backed by a self-hosted SearXNG instance."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

_TOOL_NAME = "web_search"
_SEARXNG_BASE_URL_ENV = "SEARXNG_BASE_URL"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RESULTS = 5
_DEFAULT_LANGUAGE = "auto"
_DEFAULT_SAFESEARCH = 1


def _json_error(message: str, *, query: str) -> str:
    return json.dumps({"error": message, "query": query}, ensure_ascii=False)


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


def _normalize_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    stripped = base_url.rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return stripped


def _normalize_result(raw: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(raw.get("title") or ""),
        "url": str(raw.get("url") or ""),
        "content": str(raw.get("content") or raw.get("snippet") or ""),
    }


def _build_params(
    query: str,
    max_results: int,
    *,
    language: str,
    safesearch: int,
    categories: str | None,
    engines: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "language": language,
        "safesearch": safesearch,
    }
    if categories:
        params["categories"] = categories
    if engines:
        params["engines"] = engines
    # SearXNG does not consistently honor result count across engines, but some
    # deployments forward this parameter. We still slice locally after parsing.
    params["pageno"] = 1
    params["count"] = max_results
    return params


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> str:
    """Search the web using a self-hosted SearXNG instance.

    Use this tool to find current information, news, articles, and facts from the internet.

    Args:
        query: Search keywords describing what you want to find. Be specific for better results.
        max_results: Maximum number of results to return. Default is 5.
    """
    extra = _config_extra()
    base_url = _normalize_base_url(_string_from_config(extra, "base_url", _SEARXNG_BASE_URL_ENV))
    if base_url is None:
        return _json_error("SEARXNG_BASE_URL or web_search.base_url must be configured", query=query)

    max_results = _int_from_config(extra, "max_results", max_results, minimum=1, maximum=20)
    timeout_seconds = _float_from_config(extra, "timeout", _DEFAULT_TIMEOUT_SECONDS, minimum=1.0, maximum=120.0)
    language = _string_from_config(extra, "language") or _DEFAULT_LANGUAGE
    categories = _string_from_config(extra, "categories")
    engines = _string_from_config(extra, "engines")
    safesearch = _int_from_config(extra, "safesearch", _DEFAULT_SAFESEARCH, minimum=0, maximum=2)

    params = _build_params(
        query,
        max_results,
        language=language,
        safesearch=safesearch,
        categories=categories,
        engines=engines,
    )

    try:
        with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
            response = client.get("/search", params=params, headers={"Accept": "application/json"})
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("SearXNG returned HTTP %s: %s", exc.response.status_code, exc.response.text)
        return _json_error(f"SearXNG API error: HTTP {exc.response.status_code}", query=query)
    except Exception as exc:
        logger.error("SearXNG search failed: %s: %s", type(exc).__name__, exc)
        return _json_error(str(exc), query=query)

    raw_results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(raw_results, list) or not raw_results:
        return _json_error("No results found", query=query)

    normalized_results = [_normalize_result(result) for result in raw_results[:max_results] if isinstance(result, dict)]
    if not normalized_results:
        return _json_error("No results found", query=query)

    output = {
        "query": query,
        "total_results": len(normalized_results),
        "results": normalized_results,
    }
    return json.dumps(output, indent=2, ensure_ascii=False)
