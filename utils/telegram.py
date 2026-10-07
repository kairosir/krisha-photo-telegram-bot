from pathlib import Path
from typing import Sequence

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import FSInputFile, InputMediaDocument, InputMediaPhoto

MEDIA_GROUP_LIMIT = 10


def _chunks(paths: Sequence[Path], size: int = MEDIA_GROUP_LIMIT) -> list[Sequence[Path]]:
    return [paths[index : index + size] for index in range(0, len(paths), size)]


async def send_photos(bot: Bot, chat_id: int, paths: Sequence[Path]) -> None:
    for group in _chunks(paths):
        try:
            if len(group) == 1:
                await bot.send_photo(chat_id, FSInputFile(group[0]))
                continue
            media = [InputMediaPhoto(media=FSInputFile(path)) for path in group]
            await bot.send_media_group(chat_id, media=media)
        except TelegramBadRequest:
            # Telegram может отклонить фото из-за формата, размера или геометрии.
            # Документ сохраняет исходное качество и подходит для таких случаев.
            await send_documents(bot, chat_id, group, caption="Результат без сжатия")


async def send_documents(
    bot: Bot,
    chat_id: int,
    paths: Sequence[Path],
    *,
    caption: str | None = None,
) -> None:
    for group_index, group in enumerate(_chunks(paths)):
        if len(group) == 1:
            await bot.send_document(
                chat_id,
                FSInputFile(group[0]),
                caption=caption if group_index == 0 else None,
            )
            continue
        media = [
            InputMediaDocument(
                media=FSInputFile(path),
                caption=caption if group_index == 0 and index == 0 else None,
            )
            for index, path in enumerate(group)
        ]
        await bot.send_media_group(chat_id, media=media)
