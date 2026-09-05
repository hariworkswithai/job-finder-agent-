"""OpenRouter integration for chat and web search.

Uses OpenAI-compatible /chat/completions API with optional web search.
Configuration via environment / .env:
    OPENROUTER_API_KEY  (required)
    OPENROUTER_MODEL    (default: openai/gpt-4o-mini)
    OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
"""

from __future__ import annotations

import json
import os
import re
import time

import httpx

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 90.0

_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class OpenRouterError(Exception):
    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not self.api_key or self.api_key in ("your_key_here", "sk-or-v1-placeholder"):
            raise OpenRouterError(
                "config",
                "OPENROUTER_API_KEY is not set. Add it to backend/.env and restart the server.",
            )
        self.model = (model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://job-finder-agent.local",
                "X-Title": "Job Finder Agent",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        """Send a chat request and return the raw text reply."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return await self._post_text(payload)

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        validator=None,
    ) -> dict:
        """Send a chat request and return a parsed JSON object."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_err: Exception | None = None
        attempt = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
            except httpx.TimeoutException:
                last_err = OpenRouterError("timeout", "OpenRouter request timed out.")
                _backoff(attempt)
                continue
            except httpx.HTTPError as exc:
                last_err = OpenRouterError("api", f"OpenRouter connection error: {exc}")
                _backoff(attempt)
                continue

            if resp.status_code in _RETRYABLE:
                retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                last_err = OpenRouterError(
                    "rate_limit" if resp.status_code == 429 else "api",
                    f"OpenRouter responded HTTP {resp.status_code}.",
                    resp.status_code,
                )
                if attempt < MAX_RETRIES:
                    _backoff(attempt, retry_after)
                    continue
                raise last_err

            if resp.status_code != 200:
                raise OpenRouterError(
                    "api",
                    f"OpenRouter responded HTTP {resp.status_code}: {resp.text[:300]}",
                    resp.status_code,
                )

            data = _extract_json_content(resp)
            if data is None:
                last_err = OpenRouterError("parse", "OpenRouter returned no usable JSON.")
                _backoff(attempt)
                continue

            if validator is not None:
                try:
                    data = validator(data)
                except ValueError as exc:
                    last_err = OpenRouterError("invalid", f"Invalid structured response: {exc}")
                    _backoff(attempt)
                    continue
            return data

        raise last_err or OpenRouterError("api", "OpenRouter request failed.")

    async def _post_text(self, payload: dict) -> str:
        last_err: Exception | None = None
        attempt = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
            except httpx.TimeoutException:
                last_err = OpenRouterError("timeout", "OpenRouter request timed out.")
                _backoff(attempt)
                continue
            except httpx.HTTPError as exc:
                last_err = OpenRouterError("api", f"OpenRouter connection error: {exc}")
                _backoff(attempt)
                continue

            if resp.status_code in _RETRYABLE:
                retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                last_err = OpenRouterError(
                    "rate_limit" if resp.status_code == 429 else "api",
                    f"OpenRouter responded HTTP {resp.status_code}.",
                    resp.status_code,
                )
                if attempt < MAX_RETRIES:
                    _backoff(attempt, retry_after)
                    continue
                raise last_err

            if resp.status_code != 200:
                raise OpenRouterError(
                    "api",
                    f"OpenRouter responded HTTP {resp.status_code}: {resp.text[:300]}",
                    resp.status_code,
                )

            try:
                body = resp.json()
                content = body["choices"][0]["message"]["content"]
                if isinstance(content, str) and content.strip():
                    return content.strip()
            except (KeyError, IndexError, TypeError, ValueError):
                pass
            last_err = OpenRouterError("parse", "OpenRouter returned no usable text.")
            _backoff(attempt)

        raise last_err or OpenRouterError("api", "OpenRouter request failed.")

    async def web_search(
        self,
        query: str,
        *,
        max_results: int = 10,
    ) -> list[dict]:
        """Perform a web search using OpenRouter's server-side web search.

        Returns a list of blocks shaped as {"title", "url", "content"} so the
        agent can pass both the synthesized answer and the raw search results
        (with real URLs) into job extraction.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Search the web for: {query}. "
                        "Report the raw search results exactly as returned, "
                        "including the full URL for every result."
                    ),
                }
            ],
            "tools": [
                {
                    "type": "openrouter:web_search",
                    "parameters": {
                        "search_prompt": query,
                        "max_results": max_results,
                    },
                }
            ],
            "max_tokens": 2800,
        }

        last_err: Exception | None = None
        attempt = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
            except httpx.TimeoutException:
                last_err = OpenRouterError("timeout", "Web search timed out.")
                _backoff(attempt)
                continue
            except httpx.HTTPError as exc:
                last_err = OpenRouterError("api", f"Web search connection error: {exc}")
                _backoff(attempt)
                continue

            if resp.status_code in _RETRYABLE:
                retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                last_err = OpenRouterError(
                    "rate_limit" if resp.status_code == 429 else "api",
                    f"OpenRouter responded HTTP {resp.status_code}.",
                    resp.status_code,
                )
                if attempt < MAX_RETRIES:
                    _backoff(attempt, retry_after)
                    continue
                raise last_err

            if resp.status_code != 200:
                raise OpenRouterError(
                    "api",
                    f"Web search HTTP {resp.status_code}: {resp.text[:300]}",
                    resp.status_code,
                )

            try:
                body = resp.json()
                msg = body["choices"][0]["message"]
                content = msg.get("content")
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            parts.append(str(block.get("text") or block.get("content") or ""))
                    content = "\n".join(parts)
                content = content if isinstance(content, str) else str(content)

                blocks = _search_blocks_from_response(content, msg, body)
                if blocks and any(b.get("url") for b in blocks):
                    return blocks
                last_err = OpenRouterError(
                    "parse", "Web search returned no usable results or URLs."
                )
                _backoff(attempt)
            except (KeyError, IndexError, TypeError, ValueError):
                last_err = OpenRouterError("parse", "Could not parse web search response.")
                _backoff(attempt)
                continue

        raise last_err or OpenRouterError("api", "Web search failed.")


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


_MD_LINK_RE = re.compile(r"\[([^\]]{1,120})\]\((https?://[^\s)\]]{1,300})\)")
_URLS_RE = re.compile(r"https?://[^\s\"'<>)\]>,;]+")


def _search_blocks_from_response(content: str, msg: dict, body: dict) -> list[dict]:
    """Build search-result blocks from an OpenRouter web-search response.

    Sources of URLs, in priority order:
    1. `annotations[].url_citation` (server-tool / web plugin responses)
    2. `message.citations` / `body.citations`
    3. Markdown links and bare URLs embedded in the synthesized content.
    """
    results: list[dict] = []
    seen: set[str] = set()

    def _push(url, title="", snippet=""):
        url = (url or "").strip()
        if not url or url in seen:
            return
        seen.add(url)
        results.append({
            "title": _clip(title, 120) or "",
            "url": _clip(url, 300),
            "content": _clip(snippet, 3000) or "",
        })

    annotations = msg.get("annotations") or []
    for group in annotations:
        items = group if isinstance(group, list) else [group]
        for ann in items:
            if not isinstance(ann, dict):
                continue
            citation = ann.get("url_citation")
            if isinstance(citation, dict) and citation.get("url"):
                _push(
                    citation.get("url"),
                    citation.get("title"),
                    citation.get("content") or citation.get("snippet"),
                )

    for cited in (msg.get("citations") or body.get("citations") or []):
        if isinstance(cited, dict) and cited.get("url"):
            _push(
                cited.get("url"),
                cited.get("title"),
                cited.get("content") or cited.get("snippet"),
            )

    if content:
        for title, url in _MD_LINK_RE.findall(content):
            _push(url, title, "")
        for url in _URLS_RE.findall(content):
            _push(url.rstrip(".,"), "", "")

    blocks: list[dict] = []
    if content and content.strip():
        blocks.append({"title": "Search Summary", "url": "", "content": _clip(content, 12000)})
    blocks.extend(results)
    if not blocks:
        return [{"title": "Search Results", "url": "", "content": _clip(content, 12000)}]
    return blocks


def _clip(text, n: int) -> str:
    if not text:
        return ""
    text = str(text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _backoff(attempt: int, retry_after: float | None = None) -> None:
    delay = retry_after if retry_after is not None else 1.5 * (attempt ** 2)
    time.sleep(min(delay, 10))


def _extract_json_content(resp: httpx.Response) -> dict | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    try:
        content = body["choices"][0]["message"]["content"]
        if content is None:
            content = body["choices"][0]["message"].get("reasoning")
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    return try_parse_json(content)


_JSON_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def try_parse_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    attempts = [text]
    fenced = _JSON_FENCE_RE.findall(text)
    attempts.extend(fenced)
    if not fenced:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            attempts.append(text[start: end + 1])
    for candidate in attempts:
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None
