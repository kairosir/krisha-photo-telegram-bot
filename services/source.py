import asyncio
import json
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup

from config import Settings
from utils.validators import validate_listing_url

IMAGE_EXTENSION_RE = re.compile(r"\.(?:jpe?g|png|webp|avif)(?:$|[?#])", re.IGNORECASE)
URL_IN_TEXT_RE = re.compile(
    r"https?:\\?/\\?/[^\s\"'<>]+?(?:jpe?g|png|webp|avif)(?:\\?[^\s\"'<>]*)?",
    re.IGNORECASE,
)
IMAGE_KEYS = {
    "image", "images", "imageurl", "image_url", "contenturl", "content_url",
    "full", "fullscreen", "original", "photo", "photos", "src", "url",
}
DROP_QUERY_KEYS = {"w", "h", "width", "height", "resize", "quality", "q", "thumbnail"}
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class SourceError(RuntimeError):
    pass


class ListingUnavailableError(SourceError):
    pass


class AccessRestrictedError(SourceError):
    pass


def _looks_like_image_url(value: str) -> bool:
    value = value.replace("\\/", "/").strip()
    if not value or value.startswith("data:"):
        return False
    parsed = urlparse(value if not value.startswith("//") else f"https:{value}")
    host = (parsed.hostname or "").lower()
    return bool(
        IMAGE_EXTENSION_RE.search(value)
        or (host.endswith("krisha.kz") and any(part in parsed.path.lower() for part in ("image", "photo")))
    )


def _collect_json_images(node: Any, output: list[str], parent_key: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _collect_json_images(value, output, key.lower().replace("-", "_"))
    elif isinstance(node, list):
        for value in node:
            _collect_json_images(value, output, parent_key)
    elif isinstance(node, str) and (
        parent_key.replace("_", "") in {key.replace("_", "") for key in IMAGE_KEYS}
        or _looks_like_image_url(node)
    ):
        if _looks_like_image_url(node):
            output.append(node)


def _srcset_largest(srcset: str) -> str | None:
    candidates: list[tuple[float, str]] = []
    for item in srcset.split(","):
        parts = item.strip().split()
        if not parts:
            continue
        score = 1.0
        if len(parts) > 1:
            descriptor = parts[-1].lower()
            try:
                score = float(descriptor[:-1]) if descriptor[-1:] in {"w", "x"} else 1.0
            except ValueError:
                pass
        candidates.append((score, parts[0]))
    return max(candidates, default=(0.0, ""))[1] or None


def extract_image_urls(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    found: list[str] = []
    for script in soup.find_all("script"):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        script_type = (script.get("type") or "").lower()
        if "json" in script_type or script.get("id") in {"__NEXT_DATA__", "__NUXT_DATA__"}:
            try:
                _collect_json_images(json.loads(raw), found)
            except (json.JSONDecodeError, RecursionError):
                pass
        found.extend(match.replace("\\/", "/") for match in URL_IN_TEXT_RE.findall(raw))

    for meta in soup.find_all("meta"):
        key = str(meta.get("property") or meta.get("name") or "").lower()
        content = meta.get("content")
        if content and ("image" in key or _looks_like_image_url(content)):
            found.append(content)
    for link in soup.find_all("link"):
        href = link.get("href")
        if href and (str(link.get("as") or "").lower() == "image" or _looks_like_image_url(href)):
            found.append(href)
    for tag in soup.find_all(["img", "source"]):
        for attribute in ("data-full", "data-original", "data-src", "src"):
            if value := tag.get(attribute):
                found.append(value)
        for attribute in ("srcset", "data-srcset"):
            value = tag.get(attribute)
            if value and (largest := _srcset_largest(value)):
                found.append(largest)

    result: list[str] = []
    seen: set[str] = set()
    for raw_url in found:
        clean = str(raw_url).replace("\\/", "/").replace("&amp;", "&").strip()
        parsed = urlparse(urljoin(page_url, clean))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            continue
        query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in DROP_QUERY_KEYS]
        canonical = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", urlencode(query), ""))
        key = canonical.lower()
        if _looks_like_image_url(canonical) and key not in seen:
            seen.add(key)
            result.append(canonical)
    return result


class RetryingHttpClient:
    def __init__(self, session: aiohttp.ClientSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings

    async def request(self, url: str) -> aiohttp.ClientResponse:
        last_error: Exception | None = None
        for attempt in range(self._settings.http_retries):
            try:
                response = await self._session.get(url, allow_redirects=True)
                if response.status in RETRYABLE_STATUSES and attempt + 1 < self._settings.http_retries:
                    response.release()
                    await asyncio.sleep(min(2**attempt, 4))
                    continue
                return response
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt + 1 < self._settings.http_retries:
                    await asyncio.sleep(min(2**attempt, 4))
        raise SourceError(f"HTTP-запрос не удался: {last_error}") from last_error


class ListingSource:
    def __init__(self, http: RetryingHttpClient, settings: Settings) -> None:
        self._http = http
        self._settings = settings

    async def get_image_urls(self, listing_url: str) -> list[str]:
        validated = validate_listing_url(listing_url)
        if validated is None:
            raise ListingUnavailableError("Некорректный URL объявления")
        response = await self._http.request(validated)
        async with response:
            if response.status in {401, 403, 429}:
                raise AccessRestrictedError(f"Сайт вернул HTTP {response.status}")
            if response.status == 404:
                raise ListingUnavailableError("Объявление не найдено")
            if response.status >= 400:
                raise SourceError(f"Krisha.kz вернул HTTP {response.status}")
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                raise ListingUnavailableError(f"Неожиданный Content-Type: {content_type}")
            if response.content_length and response.content_length > self._settings.max_page_bytes:
                raise ListingUnavailableError("Страница объявления слишком велика")
            body = await response.content.read(self._settings.max_page_bytes + 1)
            if len(body) > self._settings.max_page_bytes:
                raise ListingUnavailableError("Страница объявления слишком велика")
            html = body.decode(response.charset or "utf-8", errors="replace")
        return extract_image_urls(html, str(response.url))[: self._settings.max_images_per_listing]
