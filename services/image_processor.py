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
    def __init__(self, semaphore: asyncio.Semaphore, inpaint_radius: float = 4.0) -> None:
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

    def _create_auto_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Пытается найти полупрозрачные / светлые водяные знаки.
        Работает лучше на типичных логотипах Krisha.
        """
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Светлые области с низкой насыщенностью (типичный полупрозрачный ватермарк)
        # H: любой, S: низкая, V: высокая
        lower = np.array([0, 0, 160])
        upper = np.array([180, 60, 255])
        mask1 = cv2.inRange(hsv, lower, upper)

        # Дополнительно ловим почти белые пиксели
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        mask2 = cv2.inRange(hsv, lower_white, upper_white)

        mask = cv2.bitwise_or(mask1, mask2)

        # Убираем слишком мелкий шум
        kernel_small = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

        # Расширяем, чтобы захватить края логотипа
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Ограничиваем область поиска — водяные знаки почти всегда внизу или в углах
        # (чтобы не портить светлые стены/небо)
        restricted = np.zeros_like(mask)
        
        # Нижняя 18% высоты
        bottom_start = int(height * 0.82)
        restricted[bottom_start:, :] = mask[bottom_start:, :]

        # Верхние углы
        corner_h = int(height * 0.12)
        corner_w = int(width * 0.30)
        restricted[:corner_h, :corner_w] = mask[:corner_h, :corner_w]          # верхний левый
        restricted[:corner_h, width - corner_w:] = mask[:corner_h, width - corner_w:]  # верхний правый

        # Нижние углы уже входят в нижнюю зону

        return restricted

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

        # Явная маска (если передали)
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

        # Автоматическое удаление
        if auto_remove_watermark and not np.any(mask):
            auto_mask = self._create_auto_mask(image)
            mask = cv2.bitwise_or(mask, auto_mask)

        if not np.any(mask):
            # Ничего не нашли — просто копируем
            shutil.copy2(source, destination)
            return destination

        # Финальное небольшое расширение
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
