import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

@dataclass(frozen=True, slots=True)
class InpaintRegion:
    """Прямоугольная область в пикселях исходного изображения."""
    x: int
    y: int
    width: int
    height: int

class ImageProcessingError(RuntimeError):
    pass

class ImageProcessor:
    def __init__(self, semaphore: asyncio.Semaphore, inpaint_radius: float = 3.0) -> None:
        self._semaphore = semaphore
        self._inpaint_radius = inpaint_radius

    async def process(
        self,
        source: Path,
        destination: Path,
        *,
        regions: Sequence[InpaintRegion] | None = None,
        mask_path: Path | None = None,
        auto_remove_watermark: bool = True,
    ) -> Path:
        """Сохраняет оригинал либо применяет OpenCV inpainting по маске/областям."""
        async with self._semaphore:
            return await asyncio.to_thread(
                self._process_sync,
                source,
                destination,
                regions or (),
                mask_path,
                auto_remove_watermark,
            )

    def _detect_krisha_watermark_regions(self, width: int, height: int) -> list[InpaintRegion]:
        """Типичные зоны водяного знака на фото Krisha.kz."""
        regions: list[InpaintRegion] = []

        # Нижний правый угол (самый частый)
        w = max(120, int(width * 0.28))
        h = max(40, int(height * 0.09))
        regions.append(InpaintRegion(
            x=width - w - 8,
            y=height - h - 8,
            width=w,
            height=h,
        ))

        # Нижний центр
        w2 = max(160, int(width * 0.35))
        h2 = max(35, int(height * 0.07))
        regions.append(InpaintRegion(
            x=(width - w2) // 2,
            y=height - h2 - 6,
            width=w2,
            height=h2,
        ))

        # Верхний правый
        regions.append(InpaintRegion(
            x=width - w - 8,
            y=8,
            width=w,
            height=h,
        ))

        return regions

    def _process_sync(
        self,
        source: Path,
        destination: Path,
        regions: Sequence[InpaintRegion],
        mask_path: Path | None,
        auto_remove_watermark: bool,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageProcessingError(f"Не удалось открыть изображение: {source.name}")

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Явная маска
        if mask_path is not None:
            with Image.open(mask_path) as pil_mask:
                pil_mask = ImageOps.grayscale(pil_mask).resize((width, height), Image.Resampling.NEAREST)
                mask = np.asarray(pil_mask, dtype=np.uint8)
                mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        # Явные регионы
        for region in regions:
            x1 = max(0, region.x)
            y1 = max(0, region.y)
            x2 = min(width, region.x + max(0, region.width))
            y2 = min(height, region.y + max(0, region.height))
            if x1 < x2 and y1 < y2:
                mask[y1:y2, x1:x2] = 255

        # Автоматическое удаление типичных водяных знаков Krisha
        if auto_remove_watermark and not np.any(mask):
            for region in self._detect_krisha_watermark_regions(width, height):
                x1 = max(0, region.x)
                y1 = max(0, region.y)
                x2 = min(width, region.x + region.width)
                y2 = min(height, region.y + region.height)
                if x1 < x2 and y1 < y2:
                    mask[y1:y2, x1:x2] = 255

        if not np.any(mask):
            shutil.copy2(source, destination)
            return destination

        # Расширяем маску, чтобы захватить края
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        result = cv2.inpaint(image, mask, self._inpaint_radius, cv2.INPAINT_TELEA)

        suffix = destination.suffix.lower()
        params: list[int] = []
        if suffix in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif suffix == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

        if not cv2.imwrite(str(destination), result, params):
            raise ImageProcessingError(f"Не удалось сохранить изображение: {destination.name}")

        return destination
