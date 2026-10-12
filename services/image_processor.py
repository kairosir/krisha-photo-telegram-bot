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

    # ------------------------------------------------------------------
    # Автоматическое обнаружение водяных знаков (полнокадровое)
    # ------------------------------------------------------------------

    @staticmethod
    def _colorfulness(image: np.ndarray) -> np.ndarray:
        """Карта «бесцветности» (0..255). Водяные знаки Krisha — белые/серые, у них
        почти нет насыщения; цветной контент (трава, мебель, небо) получает высокий балл."""
        b = image[:, :, 0].astype(np.int16)
        g = image[:, :, 1].astype(np.int16)
        r = image[:, :, 2].astype(np.int16)
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        sat = (maxc - minc).astype(np.float32)
        sat /= np.maximum(maxc.astype(np.float32), 1.0)
        return np.clip(sat * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _multiscale_response(gray_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Отклики по гауссовым разностям на мелком (σ=1.5) и крупном (σ=24) масштабах.
        Мелкий ловит тонкие штрихи букв, крупный — мягкие ореолы полупрозрачных
        тайлинговых знаков; максимум по всем σ размывает границу между ними,
        поэтому используется пара «узкий + широкий»."""
        fine = np.abs(gray_f - cv2.GaussianBlur(gray_f, (0, 0), 1.5))
        wide = np.abs(gray_f - cv2.GaussianBlur(gray_f, (0, 0), 24.0))
        resp = np.maximum(fine, wide)
        return resp, fine

    def _detect_watermarks(self, image: np.ndarray) -> np.ndarray:
        """Сканирует ВСЁ изображение целиком без привязки к зонам (углам/центру).
        Возвращает маску пикселей, похожих на текстовый/логотипный водяной знак."""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)

        resp, fine = self._multiscale_response(gray_f)

        # Адаптивный порог относительно шума фона этого конкретного кадра.
        # Считаем его по «тихой половине» кадра (ровные области), иначе кадр,
        # густо покрытый тайлинговым знаком, поднял бы порог и знак бы пропал.
        p50 = float(np.percentile(resp, 50.0))
        thr = max(6.0, min(p50 * 4.0 + 1.5, 22.0))
        candidates = (resp > thr).astype(np.uint8) * 255

        # Хроматическая фильтрация: оставляем только achromatic (белый/серый) контент.
        colorless = cv2.threshold(self._colorfulness(image), 45, 255, cv2.THRESH_BINARY_INV)[1]
        candidates = cv2.bitwise_and(candidates, colorless)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel, iterations=2)
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidates, connectivity=8
        )

        area_limit = 0.18 * width * height  # крупный логотип может занимать до ~18% кадра
        min_side = max(3, int(min(width, height) * 0.004))
        # Контраст символа: медиана мелкого отклика внутри компонента против
        # медианы широкой окрестности. Коэффициент 1.3 оставляет реальные глифы
        # (контраст знака к фону обычно ≥2) и отсекает мягкий ореол вокруг них,
        # который при сильных знаках иначе порождает ложные срабатывания на шуме.
        med_resp = cv2.medianBlur(resp.astype(np.uint8), 9).astype(np.float32)
        mask = np.zeros((height, width), dtype=np.uint8)

        def local_bg(cx: float, cy: float) -> float:
            x1, y1 = max(0, int(cx - 40)), max(0, int(cy - 40))
            x2, y2 = min(width, int(cx + 40)), min(height, int(cy + 40))
            return float(np.median(med_resp[y1:y2, x1:x2]))

        for i in range(1, num_labels):
            _, _, w, h, area = stats[i]
            if area < min_side * min_side or w < min_side or h < min_side:
                continue
            component = labels == i
            comp_med = float(np.median(fine[component]))
            cx, cy = centroids[i]
            if comp_med < max(local_bg(cx, cy) * 1.3, thr * 0.6):
                continue
            # Форма символа/строки: разумное соотношение сторон
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 40:
                continue
            # Сверхкрупные компоненты — почти всегда сам кадр (границы зон,
            # большие ровные области); настоящий знак такого размера не бывает.
            if area > area_limit:
                continue
            mask[component] = 255

        if not mask.any():
            return mask

        # Группировка соседних символов в строки/логотипы и отсев одиночного мусора
        connect = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        grouped = cv2.dilate(mask, connect, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
        final = np.zeros_like(mask)
        min_group_area = max(60, int(0.00005 * width * height))
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_group_area:
                final[(labels == i) & (mask > 0)] = 255
        if not final.any():
            return final

        # Дорастить до реальных границ штриха (антиалиасинг букв) и добавить
        # небольшой буфер, чтобы TELEA полностью закрыл ореол вокруг глифов.
        stroke_thr = max(4.0, thr * 0.45)
        stroke = (resp > stroke_thr).astype(np.uint8) * 255
        grown = cv2.dilate(final, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        final = cv2.bitwise_or(cv2.bitwise_and(stroke, grown), final)
        final = cv2.dilate(final, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
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
            mask = cv2.bitwise_or(mask, detected)

        if not mask.any():
            # Ничего не нашли — переупаковываем без потерь качества вместо битого JPEG-перекодирования
            shutil.copy2(source, destination)
            return destination

        result = cv2.inpaint(image, mask, self._inpaint_radius, cv2.INPAINT_TELEA)

        # Итеративные проходы: остаточный ореол знака остаётся ВДОЛЬ уже
        # вычищенных границ (после TELEA контраст фона падает и детектор с
        # адаптивным порогом начинает ловить обычный шум по всему кадру).
        # Поэтому дорабатываем не глобальным детектором, а локальным порогом
        # строго в окрестности удалённых областей.
        band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        for _ in range(2):
            band = cv2.bitwise_and(
                cv2.dilate(mask, band_kernel),
                cv2.bitwise_not(cv2.dilate(mask, outer_kernel)),
            )
            if not band.any():
                break
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            resp, _ = self._multiscale_response(gray.astype(np.float32))
            colorless = cv2.threshold(self._colorfulness(result), 45, 255, cv2.THRESH_BINARY_INV)[1]
            outside = cv2.bitwise_not(cv2.dilate(mask, outer_kernel))
            ref = float(np.percentile(resp[outside > 0], 90.0)) if np.any(outside) else 12.0
            residual = ((resp > max(ref * 1.6, 8.0)) & (colorless > 0)).astype(np.uint8) * 255
            residual = cv2.morphologyEx(
                residual, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
            )
            residual = cv2.bitwise_and(residual, band)
            if not residual.any():
                break
            result = cv2.inpaint(result, residual, self._inpaint_radius, cv2.INPAINT_TELEA)
            mask = cv2.bitwise_or(mask, residual)

        self._save_result(result, alpha, destination, jpeg_quality=95)
        return destination
