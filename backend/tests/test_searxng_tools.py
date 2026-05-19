"""Unit tests for the SearXNG community web search tool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx


class TestSearXNGSearchTool:
    def test_returns_normalized_results(self):
        with patch("deerflow.community.searxng.tools.get_app_config") as mock_config:
            tool_config = MagicMock()
            tool_config.model_extra = {
                "base_url": "https://searxng.example.test",
                "max_results": 2,
                "timeout": 10,
                "language": "en",
                "safesearch": 2,
                "categories": "general",
                "engines": "duckduckgo,brave",
            }
            mock_config.return_value.get_tool_config.return_value = tool_config

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1", "content": "Snippet 1"},
                    {"title": "Result 2", "url": "https://example.com/2", "content": "Snippet 2"},
                    {"title": "Result 3", "url": "https://example.com/3", "content": "Snippet 3"},
                ]
            }
            mock_response.raise_for_status = MagicMock()

            with patch("deerflow.community.searxng.tools.httpx.Client") as mock_client_cls:
                mock_get = mock_client_cls.return_value.__enter__.return_value.get
                mock_get.return_value = mock_response

                from deerflow.community.searxng.tools import web_search_tool

                result = web_search_tool.invoke({"query": "agent browsers", "max_results": 5})
                parsed = json.loads(result)

        assert parsed == {
            "query": "agent browsers",
            "total_results": 2,
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "Snippet 1"},
                {"title": "Result 2", "url": "https://example.com/2", "content": "Snippet 2"},
            ],
        }
        mock_client_cls.assert_called_once_with(base_url="https://searxng.example.test", timeout=10.0)
        params = mock_get.call_args.kwargs["params"]
        assert params == {
            "q": "agent browsers",
            "format": "json",
            "language": "en",
            "safesearch": 2,
            "categories": "general",
            "engines": "duckduckgo,brave",
            "pageno": 1,
            "count": 2,
        }
        assert mock_get.call_args.args == ("/search",)

    def test_uses_env_base_url_when_config_omits_it(self):
        with patch("deerflow.community.searxng.tools.get_app_config") as mock_config:
            tool_config = MagicMock()
            tool_config.model_extra = {"max_results": 1}
            mock_config.return_value.get_tool_config.return_value = tool_config

            mock_response = MagicMock()
            mock_response.json.return_value = {"results": [{"title": "T", "url": "https://x.test", "content": "C"}]}
            mock_response.raise_for_status = MagicMock()

            with patch.dict("os.environ", {"SEARXNG_BASE_URL": "https://env-searxng.test"}):
                with patch("deerflow.community.searxng.tools.httpx.Client") as mock_client_cls:
                    mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response

                    from deerflow.community.searxng.tools import web_search_tool

                    result = web_search_tool.invoke({"query": "q"})
                    parsed = json.loads(result)

        assert parsed["total_results"] == 1
        mock_client_cls.assert_called_once_with(base_url="https://env-searxng.test", timeout=30.0)

    def test_missing_base_url_returns_error_json(self):
        with patch("deerflow.community.searxng.tools.get_app_config") as mock_config:
            tool_config = MagicMock()
            tool_config.model_extra = {}
            mock_config.return_value.get_tool_config.return_value = tool_config

            with patch.dict("os.environ", {}, clear=True):
                from deerflow.community.searxng.tools import web_search_tool

                result = web_search_tool.invoke({"query": "q"})
                parsed = json.loads(result)

        assert "SEARXNG_BASE_URL" in parsed["error"]
        assert parsed["query"] == "q"

    def test_http_error_returns_structured_error(self):
        with patch("deerflow.community.searxng.tools.get_app_config") as mock_config:
            tool_config = MagicMock()
            tool_config.model_extra = {"base_url": "https://searxng.example.test"}
            mock_config.return_value.get_tool_config.return_value = tool_config

            response = MagicMock()
            response.status_code = 403
            response.text = "Forbidden"
            error = httpx.HTTPStatusError("403", request=MagicMock(), response=response)

            with patch("deerflow.community.searxng.tools.httpx.Client") as mock_client_cls:
                mock_get = mock_client_cls.return_value.__enter__.return_value.get
                mock_get.side_effect = error

                from deerflow.community.searxng.tools import web_search_tool

                result = web_search_tool.invoke({"query": "q"})
                parsed = json.loads(result)

        assert parsed == {"error": "SearXNG API error: HTTP 403", "query": "q"}

    def test_empty_results_returns_error_json(self):
        with patch("deerflow.community.searxng.tools.get_app_config") as mock_config:
            tool_config = MagicMock()
            tool_config.model_extra = {"base_url": "https://searxng.example.test"}
            mock_config.return_value.get_tool_config.return_value = tool_config

            mock_response = MagicMock()
            mock_response.json.return_value = {"results": []}
            mock_response.raise_for_status = MagicMock()

            with patch("deerflow.community.searxng.tools.httpx.Client") as mock_client_cls:
                mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response

                from deerflow.community.searxng.tools import web_search_tool

                result = web_search_tool.invoke({"query": "nothing"})
                parsed = json.loads(result)

        assert parsed == {"error": "No results found", "query": "nothing"}

    def test_snippet_fallback_and_missing_fields(self):
        with patch("deerflow.community.searxng.tools.get_app_config") as mock_config:
            tool_config = MagicMock()
            tool_config.model_extra = {"base_url": "https://searxng.example.test", "max_results": 2}
            mock_config.return_value.get_tool_config.return_value = tool_config

            mock_response = MagicMock()
            mock_response.json.return_value = {"results": [{"snippet": "Fallback snippet"}, {}]}
            mock_response.raise_for_status = MagicMock()

            with patch("deerflow.community.searxng.tools.httpx.Client") as mock_client_cls:
                mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_response

                from deerflow.community.searxng.tools import web_search_tool

                result = web_search_tool.invoke({"query": "partial"})
                parsed = json.loads(result)

        assert parsed["results"] == [
            {"title": "", "url": "", "content": "Fallback snippet"},
            {"title": "", "url": "", "content": ""},
        ]
