import asyncio
import logging
from pathlib import Path

import aiohttp
from aiogram import Bot, F, Router
from aiogram.types import Message

from config import Settings
from services.database import Database
from services.image_processor import ImageProcessor
from services.krisha import (
    AccessRestrictedError,
    KrishaClient,
    KrishaError,
    ListingUnavailableError,
    validate_listing_url,
)
from utils.files import temporary_directory
from utils.telegram import send_documents, send_photos

router = Router(name="listings")
logger = logging.getLogger(__name__)

_active_tasks: dict[int, asyncio.Task[object]] = {}
_tasks_guard = asyncio.Lock()
_processing_semaphore: asyncio.Semaphore | None = None


def cancel_user_task(user_id: int) -> bool:
    task = _active_tasks.get(user_id)
    if task and not task.done():
        task.cancel()
        return True
    return False


def _get_processing_semaphore(limit: int) -> asyncio.Semaphore:
    global _processing_semaphore
    if _processing_semaphore is None:
        _processing_semaphore = asyncio.Semaphore(limit)
    return _processing_semaphore


async def _process_listing(message: Message, bot: Bot, settings: Settings, url: str) -> int:
    await message.answer("🔎 Получаю фотографии объявления…")
    timeout = aiohttp.ClientTimeout(total=settings.http_timeout_seconds)
    headers = {
        "User-Agent": settings.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,image/*;q=0.8,*/*;q=0.5",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.6",
    }

    async with temporary_directory(prefix="krisha-bot-") as work_dir:
        connector = aiohttp.TCPConnector(limit=max(settings.download_concurrency * 2, 10))
        async with aiohttp.ClientSession(timeout=timeout, headers=headers, connector=connector) as session:
            client = KrishaClient(session=session, settings=settings)
            image_urls = await client.get_image_urls(url)
            if not image_urls:
                raise ListingUnavailableError("На странице не найдены фотографии")

            originals_dir = work_dir / "originals"
            processed_dir = work_dir / "processed"
            originals_dir.mkdir()
            processed_dir.mkdir()

            originals = await client.download_images(image_urls, originals_dir)
            if not originals:
                raise ListingUnavailableError("Не удалось скачать подходящие фотографии")

            await message.answer(
                f"📸 Найдено фотографий: {len(originals)}\n⏳ Обрабатываю…"
            )
            processor = ImageProcessor(_get_processing_semaphore(settings.global_processing_limit))
            processed: list[Path] = []
            for index, image in enumerate(originals, start=1):
                output = processed_dir / f"processed_{index:03d}{image.path.suffix.lower()}"
                processed.append(await processor.process(image.path, output))

            await send_photos(bot, message.chat.id, processed)
            if settings.send_originals_as_files:
                await message.answer("📎 Отправляю оригиналы без сжатия как файлы…")
                await send_documents(
                    bot,
                    message.chat.id,
                    [item.path for item in originals],
                    caption="Оригиналы без сжатия",
                )
            await message.answer(f"✅ Готово!\nОбработано фотографий: {len(processed)}")
            return len(processed)


@router.message(F.text)
async def listing_handler(
    message: Message,
    bot: Bot,
    settings: Settings,
    database: Database,
) -> None:
    if not message.from_user or not message.text:
        return
    url = validate_listing_url(message.text.strip())
    if url is None:
        await message.answer(
            "❌ Не удалось распознать ссылку. Отправь ссылку на объявление Krisha.kz."
        )
        return

    user_id = message.from_user.id
    async with _tasks_guard:
        existing = _active_tasks.get(user_id)
        if existing and not existing.done():
            await message.answer(
                "⏳ Предыдущая задача ещё выполняется. Дождись результата или используй /cancel."
            )
            return
        task = asyncio.current_task()
        if task is not None:
            _active_tasks[user_id] = task

    job_id = await database.create_job(user_id, url)
    try:
        processed_count = await _process_listing(message, bot, settings, url)
        await database.complete_job(job_id, processed_count)
    except asyncio.CancelledError:
        await database.cancel_job(job_id)
        logger.info("Задача пользователя %s отменена", user_id)
        raise
    except (AccessRestrictedError, ListingUnavailableError) as exc:
        await database.fail_job(job_id, str(exc))
        logger.warning("Объявление недоступно для пользователя %s: %s", user_id, exc)
        await message.answer(
            "❌ Не удалось получить фотографии объявления. Возможно, объявление удалено "
            "или сайт ограничил автоматический доступ."
        )
    except KrishaError as exc:
        await database.fail_job(job_id, str(exc))
        logger.warning("Ошибка Krisha.kz для пользователя %s: %s", user_id, exc)
        await message.answer("❌ Ошибка при загрузке фотографий. Попробуй ещё раз позже.")
    except Exception as exc:
        await database.fail_job(job_id, str(exc))
        logger.exception("Необработанная ошибка для пользователя %s", user_id)
        await message.answer("❌ Произошла внутренняя ошибка. Попробуй ещё раз позже.")
    finally:
        async with _tasks_guard:
            if _active_tasks.get(user_id) is asyncio.current_task():
                _active_tasks.pop(user_id, None)


@router.message()
async def unsupported_handler(message: Message) -> None:
    await message.answer("Отправь текстовую ссылку на объявление Krisha.kz.")
