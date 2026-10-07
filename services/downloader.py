import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

from config import Settings
from services.source import RetryingHttpClient, SourceError

logger = logging.getLogger(__name__)


class ImageDownloadError(SourceError):
    pass


@dataclass(frozen=True, slots=True)
class DownloadedImage:
    path: Path
    source_url: str
    sha256: str
    width: int
    height: int


class ImageDownloader:
    def __init__(self, http: RetryingHttpClient, settings: Settings) -> None:
        self._http = http
        self._settings = settings

    async def download_images(
        self,
        urls: Iterable[str],
        directory: Path,
    ) -> list[DownloadedImage]:
        semaphore = asyncio.Semaphore(self._settings.download_concurrency)

        async def download(index: int, url: str) -> DownloadedImage | None:
            async with semaphore:
                try:
                    return await self._download_one(index, url, directory)
                except (ImageDownloadError, UnidentifiedImageError, OSError) as exc:
                    logger.info("Изображение пропущено (%s): %s", url, exc)
                    return None

        results = await asyncio.gather(*(download(i, url) for i, url in enumerate(urls, 1)))
        unique: list[DownloadedImage] = []
        hashes: set[str] = set()
        for item in results:
            if item is None:
                continue
            if item.sha256 in hashes:
                item.path.unlink(missing_ok=True)
                continue
            hashes.add(item.sha256)
            unique.append(item)
        return unique

    async def _download_one(self, index: int, url: str, directory: Path) -> DownloadedImage:
        response = await self._http.request(url)
        async with response:
            if response.status >= 400:
                raise ImageDownloadError(f"HTTP {response.status}")
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if not content_type.startswith("image/"):
                raise ImageDownloadError(f"Content-Type не является изображением: {content_type}")
            if response.content_length and response.content_length > self._settings.max_image_bytes:
                raise ImageDownloadError("Файл превышает допустимый размер")

            suffix_by_type = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
                "image/avif": ".avif",
            }
            suffix = suffix_by_type.get(
                content_type,
                Path(urlparse(url).path).suffix.lower() or ".img",
            )
            destination = directory / f"image_{index:03d}{suffix}"
            digest = hashlib.sha256()
            total = 0
            try:
                with destination.open("wb") as output:
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        total += len(chunk)
                        if total > self._settings.max_image_bytes:
                            raise ImageDownloadError("Файл превышает допустимый размер")
                        digest.update(chunk)
                        output.write(chunk)
            except Exception:
                destination.unlink(missing_ok=True)
                raise

        try:
            with Image.open(destination) as image:
                image.verify()
            with Image.open(destination) as image:
                width, height = image.size
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        if width < self._settings.min_image_width or height < self._settings.min_image_height:
            destination.unlink(missing_ok=True)
            raise ImageDownloadError(f"Миниатюра {width}x{height}")
        return DownloadedImage(destination, url, digest.hexdigest(), width, height)
