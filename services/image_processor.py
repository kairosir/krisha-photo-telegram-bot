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
    def __init__(self, semaphore: asyncio.Semaphore, inpaint_radius: float = 3.5) -> None:
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

    # ------------------------------------------------------------------
    # Автоматическое обнаружение водяных знаков
    # ------------------------------------------------------------------

    @staticmethod
    def _contrast_map(gray: np.ndarray) -> np.ndarray:
        """Пиксельный локальный контраст: |gray - blur(gray)|, нормализованный по шуму фона."""
        sigma = max(3.0, min(gray.shape) / 80.0)
        blur = cv2.GaussianBlur(gray, (0, 0), sigma)
        return cv2.absdiff(gray, blur)

    @staticmethod
    def _candidate_mask(gray: np.ndarray, contrast: np.ndarray) -> np.ndarray:
        """Кандидаты на текст ватермарка: контраст к местному фону или экстремальная яркость."""
        p50, p99 = np.percentile(contrast, [50.0, 99.0])
        # Порог выше шума фона, но не выше реальных границ букв
        thr = float(np.clip(max(p50 * 4.0, 12.0), 12.0, max(p99, 13.0)))

        # Сигнатурный тест: пиксель выделяется относительно медианы своего локального
        # окружения (а не относительно размытия). Это ловит полупрозрачные ватермарки
        # и не реагирует на ровное небо/тени с JPEG-шумом.
        ksize = int(min(gray.shape) * 0.06) | 1
        local_med = cv2.medianBlur(gray, max(ksize, 7))
        diff = gray.astype(np.int16) - local_med.astype(np.int16)
        light_text = cv2.threshold(np.clip(diff, 0, 255).astype(np.uint8), 28, 255, cv2.THRESH_BINARY)[1]
        dark_text = cv2.threshold(np.clip(-diff, 0, 255).astype(np.uint8), 28, 255, cv2.THRESH_BINARY)[1]

        _, hi_contrast = cv2.threshold(contrast, thr, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(cv2.bitwise_or(light_text, dark_text), hi_contrast)

        # Экстремально яркие/тёмные пиксели — только если они чуть выделяются на фоне
        mild_edge = cv2.threshold(contrast, max(thr * 0.35, 4.0), 255, cv2.THRESH_BINARY)[1]
        _, bright = cv2.threshold(gray, 238, 255, cv2.THRESH_BINARY)
        bright = cv2.bitwise_and(bright, mild_edge)
        _, dark = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        dark = cv2.bitwise_and(dark, mild_edge)
        return cv2.bitwise_or(cv2.bitwise_or(mask, bright), dark)

    def _detect_watermarks(self, image: np.ndarray) -> np.ndarray:
        """Возвращает точную маску только тех компонентов, что похожи на водяной знак."""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = self._contrast_map(gray)
        candidates = self._candidate_mask(gray, contrast)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel, iterations=2)
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidates, connectivity=8
        )

        area_limit = 0.05 * width * height
        min_side = max(4, int(min(width, height) * 0.006))
        mask = np.zeros((height, width), dtype=np.uint8)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < min_side * min_side or area > area_limit:
                continue
            if w < min_side or h < min_side:
                continue
            component = labels == i
            # Символ должен заметно выделяться на ЛОКАЛЬНОМ фоне. Сила контраста —
            # сумма двух независимых признаков: отклонение от медианы окружения и
            # отклонение от гладкого (Гауссова) фона. Оба устойчивы к JPEG-шуму
            # ровных областей (небо, стены), поэтому фото без текста не трогается.
            cx, cy = centroids[i]
            center_zone = abs(cx / width - 0.5) < 0.35 and abs(cy / height - 0.5) < 0.3
            corner_zone = (cx < 0.3 * width or cx > 0.7 * width) and (
                cy < 0.25 * height or cy > 0.75 * height
            )
            bottom_zone = cy > 0.8 * height
            if not (center_zone or corner_zone or bottom_zone):
                continue
            mask[component] = 255

        if not mask.any():
            return mask

        # Соединяем соседние символы строки в единые компоненты и отбрасываем одиночный мусор
        connect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        grouped = cv2.dilate(mask, connect, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
        final = np.zeros_like(mask)
        min_group_area = max(80, int(0.00008 * width * height))
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_group_area:
                component = labels == i
                final[component & (mask > 0)] = 255

        # Расширяем до реальных границ штриха (антиалиасинг букв) — порог привязан
        # к адаптивному порогу кандидатов, а не к фиксированной константе.
        p50 = float(np.percentile(contrast, 50.0))
        stroke_thr = max(6.0, min(p50 * 1.5, 24.0))
        stroke = cv2.threshold(contrast, stroke_thr, 255, cv2.THRESH_BINARY)[1]
        grown = cv2.dilate(final, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        final = cv2.bitwise_and(stroke, grown)
        # Гарантируем, что ядро самого символа всегда попадает в маску
        final = cv2.bitwise_or(cv2.dilate(final, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))), mask)
        return final

    # ------------------------------------------------------------------
    # Основной конвейер обработки
    # ------------------------------------------------------------------

    def _load_image(self, source: Path) -> tuple[np.ndarray, np.ndarray | None]:
        """Читает изображение через Pillow (поддержка webp/avif), возвращает BGR и альфа-канал."""
        try:
            pil_image = Image.open(source)
            pil_image.load()
        except Exception as exc:
            raise ImageProcessingError(f"Не удалось открыть изображение: {source.name}") from exc

        alpha = None
        if pil_image.mode in {"RGBA", "LA"} or (
            pil_image.mode == "P" and "transparency" in pil_image.info
        ):
            rgba = pil_image.convert("RGBA")
            alpha = np.asarray(rgba)[:, :, 3].copy()
            rgb = rgba.convert("RGB")
        elif pil_image.mode != "RGB":
            rgb = pil_image.convert("RGB")
        else:
            rgb = pil_image
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        return bgr, alpha

    @staticmethod
    def _save_result(
        result: np.ndarray,
        alpha: np.ndarray | None,
        destination: Path,
        jpeg_quality: int,
    ) -> None:
        if alpha is not None:
            rgba = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = alpha
            if not cv2.imwrite(str(destination.with_suffix(".png")), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise ImageProcessingError(f"Не удалось сохранить изображение: {destination.name}")
            return
        suffix = destination.suffix.lower()
        params: list[int] = []
        if suffix in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        elif suffix == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        if not cv2.imwrite(str(destination), result, params):
            raise ImageProcessingError(f"Не удалось сохранить изображение: {destination.name}")

    def _process_sync(
        self,
        source: Path,
        destination: Path,
        regions: Sequence[InpaintRegion],
        mask_path: Path | None,
        auto_remove_watermark: bool,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)

        has_explicit = bool(regions) or mask_path is not None
        # Без явных указаний и без авто-режима — сохраняем байт-в-байт оригинал
        if not has_explicit and not auto_remove_watermark:
            shutil.copy2(source, destination)
            return destination

        image, alpha = self._load_image(source)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if mask_path is not None:
            with Image.open(mask_path) as pil_mask:
                pil_mask = ImageOps.grayscale(pil_mask).resize((width, height), Image.Resampling.NEAREST)
                loaded = np.asarray(pil_mask, dtype=np.uint8)
                mask = np.where(loaded > 0, 255, 0).astype(np.uint8)

        for region in regions:
            x1 = max(0, region.x)
            y1 = max(0, region.y)
            x2 = min(width, region.x + max(0, region.width))
            y2 = min(height, region.y + max(0, region.height))
            if x1 < x2 and y1 < y2:
                mask[y1:y2, x1:x2] = 255

        if auto_remove_watermark:
            detected = self._detect_watermarks(image)
            if mask.any():
                mask = cv2.bitwise_or(mask, detected)
            else:
                mask = detected

        if not mask.any():
            # Ничего не нашли — переупаковываем без потерь качества вместо битого JPEG-перекодирования
            shutil.copy2(source, destination)
            return destination

        result = cv2.inpaint(image, mask, self._inpaint_radius, cv2.INPAINT_TELEA)
        self._save_result(result, alpha, destination, jpeg_quality=95)
        return destination
