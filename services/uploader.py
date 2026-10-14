import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Sequence

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.types import FSInputFile, InputMediaDocument, InputMediaPhoto

logger = logging.getLogger(__name__)

MEDIA_GROUP_LIMIT = 10


def _chunks(paths: Sequence[Path], size: int = MEDIA_GROUP_LIMIT) -> list[Sequence[Path]]:
    return [paths[index : index + size] for index in range(0, len(paths), size)]


async def stream_send_photos(
    bot: Bot,
    chat_id: int,
    paths: AsyncIterator[Path],
) -> tuple[int, list[Path]]:
    """Прогрессивная выдача: отправляет фото альбомами по 10 по мере готовности.

    Буфер пополняется из асинхронного итератора (например, результатов
    параллельной обработки); как только накоплено 10 файлов или источник
    исчерпан — альбом улетает в Telegram сразу, не дожидаясь остальных.
    Возвращает (число доставленных, список неотправленных из-за ошибок).
    """
    sent = 0
    leftovers: list[Path] = []
    buffer: list[Path] = []

    async def flush() -> None:
        nonlocal sent
        if not buffer:
            return
        group = list(buffer)
        buffer.clear()
        try:
            if len(group) == 1:
                await bot.send_photo(chat_id, FSInputFile(group[0]))
            else:
                await bot.send_media_group(
                    chat_id,
                    media=[InputMediaPhoto(media=FSInputFile(path)) for path in group],
                )
            sent += len(group)
        except TelegramBadRequest as exc:
            logger.warning("Album rejected (%s), falling back to documents", exc)
            try:
                sent += await send_documents(bot, chat_id, group, caption="Результат без сжатия")
            except TelegramAPIError as fallback_exc:
                logger.error("Fallback send failed too: %s", fallback_exc)
                leftovers.extend(group)
        except TelegramAPIError as exc:
            logger.error("send_photo group failed: %s", exc)
            leftovers.extend(group)

    async for path in paths:
        buffer.append(path)
        if len(buffer) >= MEDIA_GROUP_LIMIT:
            await flush()
    await flush()
    return sent, leftovers


async def send_photos(bot: Bot, chat_id: int, paths: Sequence[Path]) -> int:
    """Отправляет фото альбомом; при отказе Telegram — документами. Возвращает число доставленных файлов."""
    sent = 0
    for group in _chunks(paths):
        try:
            if len(group) == 1:
                await bot.send_photo(chat_id, FSInputFile(group[0]))
                sent += 1
                continue
            await bot.send_media_group(
                chat_id,
                media=[InputMediaPhoto(media=FSInputFile(path)) for path in group],
            )
            sent += len(group)
        except TelegramBadRequest as exc:
            logger.warning("Album rejected (%s), falling back to documents", exc)
            try:
                sent += await send_documents(bot, chat_id, group, caption="Результат без сжатия")
            except TelegramAPIError as fallback_exc:
                logger.error("Fallback send failed too: %s", fallback_exc)
        except TelegramAPIError as exc:
            logger.error("send_photo group failed: %s", exc)
    return sent


async def send_documents(
    bot: Bot,
    chat_id: int,
    paths: Sequence[Path],
    *,
    caption: str | None = None,
) -> int:
    """Отправляет файлы альбомами по 10; при отказе — по одному. Возвращает число доставленных файлов."""
    sent = 0
    for group_index, group in enumerate(_chunks(paths)):
        try:
            if len(group) == 1:
                await bot.send_document(
                    chat_id,
                    FSInputFile(group[0]),
                    caption=caption if group_index == 0 else None,
                )
                sent += 1
                continue
            await bot.send_media_group(
                chat_id,
                media=[
                    InputMediaDocument(
                        media=FSInputFile(path),
                        caption=caption if group_index == 0 and index == 0 else None,
                    )
                    for index, path in enumerate(group)
                ],
            )
            sent += len(group)
        except TelegramAPIError as exc:
            logger.warning("Document album %s failed (%s), sending one by one", group_index, exc)
            for index, path in enumerate(group):
                try:
                    await bot.send_document(
                        chat_id,
                        FSInputFile(path),
                        caption=caption if group_index == 0 and index == 0 else None,
                    )
                    sent += 1
                except TelegramAPIError as single_exc:
                    logger.error("Failed to send document %s: %s", path.name, single_exc)
    return sent
