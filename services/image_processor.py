import asyncio
import logging
import shutil
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

LAMA_MODEL_URLS = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
    "https://huggingface.co/cocktailpeanut/biglama/resolve/main/big-lama.pt",
)
LAMA_FILENAME = "big-lama.pt"
# Максимальный размер большей стороны кадра при нейро-инпейнте: LaMa обучена
# на ~512-768px; крупные кадры режутся на тайлы с перекрытием.
LAMA_TILE = 512


@dataclass(frozen=True, slots=True)
class InpaintRegion:
    """Прямоугольная область в пикселях исходного изображения."""
    x: int
    y: int
    width: int
    height: int


class ImageProcessingError(RuntimeError):
    pass


class _LamaEngine:
    """Ленивая singleton-обёртка над TorchScript-моделью LaMa (MIT лицензия).

    Модель скачивается один раз (~200 МБ) и держится в памяти. Инференс
    защищён lock'ом: torch сам по себе потокобезопасен только для eval,
    но память ограничена, поэтому параллелим не больше OMP threads.
    """

    _lock = threading.Lock()
    _model = None
    _device = None

    @classmethod
    def available(cls) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def _ensure_model(cls, model_dir: str | None):
        if cls._model is not None:
            return cls._model
        with cls._lock:
            if cls._model is not None:
                return cls._model
            import torch

            path = cls._locate_weights(model_dir)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.jit.load(str(path), map_location=device)
            model.eval()
            cls._model = model
            cls._device = device
            logger.info("LaMa загружена на %s (веса: %s)", device, path)
            return cls._model

    @staticmethod
    def _locate_weights(model_dir: str | None) -> Path:
        candidates: list[Path] = []
        if model_dir:
            candidates.append(Path(model_dir) / LAMA_FILENAME)
        env_dir = Path(__file__).resolve().parent.parent / "models"
        candidates.append(env_dir / LAMA_FILENAME)
        candidates.append(Path.home() / ".cache" / "krisha-bot" / LAMA_FILENAME)
        for c in candidates:
            if c.is_file():
                return c

        target = candidates[-1]
        target.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for url in LAMA_MODEL_URLS:
            try:
                logger.info("Скачиваю модель LaMa: %s", url)
                tmp = target.with_suffix(".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.rename(target)
                return target
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Не удалось скачать LaMa с %s: %s", url, exc)
        raise ImageProcessingError(
            f"Модель LaMa не найдена и не скачалась: {last_error}"
        )

    @classmethod
    def inpaint(cls, image_bgr: np.ndarray, mask: np.ndarray, model_dir: str | None) -> np.ndarray:
        """Инпейнт всего кадра. Большие фото режутся на тайлы LAMA_TILE с
        перекрытием 128px; результат склеивается линейным blending."""
        import torch

        model = cls._ensure_model(model_dir)
        device = cls._device
        height, width = image_bgr.shape[:2]

        img_f = image_bgr.astype(np.float32) / 255.0
        mask_bin = (mask > 127).astype(np.float32)
        result = img_f.copy()

        step = LAMA_TILE - 128
        tiles_x = max(1, -(-width // step)) if width > LAMA_TILE else 1
        tiles_y = max(1, -(-height // step)) if height > LAMA_TILE else 1
        if width <= LAMA_TILE and height <= LAMA_TILE:
            tiles = [(0, 0, width, height)]
        else:
            tiles = []
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    x1 = min(tx * step, max(0, width - LAMA_TILE))
                    y1 = min(ty * step, max(0, height - LAMA_TILE))
                    tiles.append((x1, y1, min(x1 + LAMA_TILE, width), min(y1 + LAMA_TILE, height)))

        pad = 8
        for (x1, y1, x2, y2) in tiles:
            tile_img = img_f[y1:y2, x1:x2]
            tile_mask = mask_bin[y1:y2, x1:x2]
            if tile_mask.max() < 0.5:
                continue
            th, tw = tile_img.shape[:2]
            padded = np.zeros((th + 2 * pad, tw + 2 * pad, 3), np.float32)
            padded[pad:pad + th, pad:pad + tw] = tile_img
            pmask = np.zeros((th + 2 * pad, tw + 2 * pad), np.float32)
            pmask[pad:pad + th, pad:pad + tw] = tile_mask
            # Выровнять по 16 для сверточных слоёв модели
            ph, pw = pmask.shape
            nh, nw = ((ph + 15) // 16) * 16, ((pw + 15) // 16) * 16
            canvas = np.zeros((nh, nw, 3), np.float32)
            canvas[:ph, :pw] = padded
            mcanvas = np.zeros((nh, nw), np.float32)
            mcanvas[:ph, :pw] = pmask

            xt = torch.from_numpy(canvas.transpose(2, 0, 1)[None]).to(device)
            mt = torch.from_numpy(mcanvas[None, None]).to(device)
            with torch.no_grad(), cls._lock:
                out = model(xt, mt)[0].permute(1, 2, 0).clamp_(0, 1).float().cpu().numpy()
            out = out[:ph, :pw, :]
            filled = out[pad:pad + th, pad:pad + tw]
            # Берём из нейросети только область маски с мягким переходом
            blend = cv2.GaussianBlur(tile_mask, (0, 0), 3.0)[..., None]
            result[y1:y2, x1:x2] = tile_img * (1 - blend) + filled * blend

        return (result * 255.0).clip(0, 255).astype(np.uint8)


class ImageProcessor:
    def __init__(
        self,
        semaphore: asyncio.Semaphore,
        inpaint_radius: float = 4.0,
        use_lama: bool = True,
        lama_model_dir: str | None = None,
    ) -> None:
        self._semaphore = semaphore
        self._inpaint_radius = inpaint_radius
        self._use_lama = use_lama and _LamaEngine.available()
        self._lama_model_dir = lama_model_dir

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

        # --- Групповой проход для тайлинговых знаков -------------------------
        # Полупрозрачные знаки, набросанные сеткой по всему кадру, дают много
        # мелких компонентов; одиночный глиф ниже min_group_area отбрасывался
        # как шум и такие кадры оставались необработанными. Если похожих на
        # символы компонентов много и они распределены по разным регионам
        # кадра регулярно — это паттерн тайлинга: строим кластерные bbox
        # (каждый повтор знака) и заполняем их всей мягкой зоной знака,
        # включая ореол, который точечная маска глифов не захватывает.
        tiling_boxes = self._tiling_boxes(labels, stats, centroids, width, height)
        if tiling_boxes is not None:
            for x, y, bw, bh in tiling_boxes:
                mask[y:y + bh, x:x + bw] = 255

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

        # Дорастить до реальных границ штриха (антиалиасинг букв). Порог берём
        # ОТНОСИТЕЛЬНО контраста самих найденных глифов: абсолютный порог
        # либо не дотягивает до мягких краёв полупрозрачного знака, либо ловит
        # шум фона. Ограничиваем доращивание окрестностью маски + ореол.
        comp_med = float(np.median(resp[mask > 0]))
        stroke_thr = max(thr * 0.45, comp_med * 0.3)
        stroke = (resp > stroke_thr).astype(np.uint8) * 255
        grown = cv2.dilate(final, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        final = cv2.bitwise_or(cv2.bitwise_and(stroke, grown), final)
        final = cv2.dilate(final, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        return final

    @staticmethod
    def _tiling_boxes(
        labels: np.ndarray,
        stats: np.ndarray,
        centroids: np.ndarray,
        width: int,
        height: int,
    ) -> list[tuple[int, int, int, int]] | None:
        """Обнаруживает регулярную сетку одинаковых символьных компонентов
        (тайлинговый водяной знак) и возвращает bbox каждого повтора со
        смягчёнными краями. None — если паттерн тайлинга не подтверждён."""
        n = labels.max()
        if n < 12:
            return None
        min_side = max(3, int(min(width, height) * 0.004))
        cand_ids = []
        for i in range(1, n + 1):
            _, _, w, h, area = stats[i]
            if area < min_side * min_side or area > 0.02 * width * height:
                continue
            if w < min_side or h < min_side or max(w, h) / max(1, min(w, h)) > 40:
                continue
            cand_ids.append(i)
        if len(cand_ids) < 12:
            return None

        pts = centroids[cand_ids]
        # Разбиваем кадр на 4x4 региона: тайлинг даёт компоненты по всей площади
        regions: set[tuple[int, int]] = set()
        for cx, cy in pts:
            regions.add((min(3, int(cx / width * 4)), min(3, int(cy / height * 4))))
        if len(regions) < 6:
            return None

        areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in cand_ids], dtype=np.float64)
        med_a = float(np.median(areas))
        if med_a <= 0:
            return None
        consistent = [i for i in cand_ids if abs(stats[i, cv2.CC_STAT_AREA] - med_a) < med_a * 0.75]
        if len(consistent) < 12:
            return None

        # Кластеризация центров повторами знака (eps ≈ медианный шаг сетки)
        pts_c = centroids[consistent]
        sizes = np.array([stats[i, cv2.CC_STAT_WIDTH] for i in consistent], dtype=np.float64)
        med_w = float(np.median(sizes)) if len(sizes) else 20.0
        eps = max(med_w * 2.5, min(width, height) * 0.05)

        assigned = np.full(len(consistent), -1, dtype=int)
        clusters: list[list[int]] = []
        for idx in range(len(consistent)):
            if assigned[idx] >= 0:
                continue
            cid = len(clusters)
            stack = [idx]
            assigned[idx] = cid
            members: list[int] = []
            while stack:
                j = stack.pop()
                members.append(j)
                near = np.where(
                    (np.abs(pts_c[:, 0] - pts_c[j, 0]) < eps)
                    & (np.abs(pts_c[:, 1] - pts_c[j, 1]) < eps * 0.8)
                    & (assigned < 0)
                )[0]
                for k in near:
                    assigned[k] = cid
                    stack.append(int(k))
            clusters.append(members)

        boxes: list[tuple[int, int, int, int]] = []
        big_enough = [c for c in clusters if len(c) >= 4]
        if len(big_enough) < 3:
            return None
        pad_x = max(8, int(med_w * 0.9))
        pad_y = max(10, int(height * 0.02))
        for members in big_enough:
            xs = pts_c[members, 0]
            ys = pts_c[members, 1]
            x1 = max(0, int(xs.min() - pad_x))
            y1 = max(0, int(ys.min() - pad_y))
            x2 = min(width, int(xs.max() + pad_x))
            y2 = min(height, int(ys.max() + pad_y))
            bw, bh = x2 - x1, y2 - y1
            # Повтор знака не должен быть больше ~35% кадра по каждой оси
            if bw > 0.35 * width or bh > 0.35 * height:
                continue
            boxes.append((x1, y1, bw, bh))
        return boxes or None

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
        # Сохраняем через Pillow: у cv2.imwrite свои дефолтные уровни качества,
        # из-за которых даже «пустая» обработка заметно портила бы JPEG.
        if alpha is not None:
            rgba = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = alpha
            pil = Image.fromarray(rgba, mode="RGBA")
            try:
                pil.save(destination.with_suffix(".png"), optimize=False, compress_level=3)
            except Exception as exc:
                raise ImageProcessingError(f"Не удалось сохранить изображение: {destination.name}") from exc
            return
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb, mode="RGB")
        suffix = destination.suffix.lower()
        try:
            if suffix in {".jpg", ".jpeg"}:
                pil.save(destination, quality=jpeg_quality, subsampling=0)
            elif suffix == ".png":
                pil.save(destination, optimize=False, compress_level=3)
            elif suffix == ".webp":
                pil.save(destination, quality=jpeg_quality, method=4)
            else:
                pil.save(destination)
        except Exception as exc:
            raise ImageProcessingError(f"Не удалось сохранить изображение: {destination.name}") from exc

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

        # Расширяем маску: полупрозрачные знаки имеют мягкие края, которые
        # детектор захватывает не полностью; без буфера LaMa оставит ореол.
        work_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        result = None
        if self._use_lama:
            try:
                result = _LamaEngine.inpaint(image, work_mask, self._lama_model_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LaMa недоступна (%s), откат на OpenCV TELEA", exc)
                self._use_lama = False
        if result is None:
            result = cv2.inpaint(image, work_mask, self._inpaint_radius, cv2.INPAINT_TELEA)

        # Итеративная доработка остатков (полезна в обоих режимах: у LaMa тоже
        # бывают пропуски на самых плотных тайлинговых знаках).
        band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        passes = 1 if self._use_lama else 2
        for _ in range(passes):
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
            if self._use_lama:
                try:
                    result = _LamaEngine.inpaint(result, residual, self._lama_model_dir)
                except Exception:  # noqa: BLE001
                    result = cv2.inpaint(result, residual, self._inpaint_radius, cv2.INPAINT_TELEA)
            else:
                result = cv2.inpaint(result, residual, self._inpaint_radius, cv2.INPAINT_TELEA)
            mask = cv2.bitwise_or(mask, residual)

        self._save_result(result, alpha, destination, jpeg_quality=95)
        return destination
