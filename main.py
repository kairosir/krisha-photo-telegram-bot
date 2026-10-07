import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from config import get_settings
from handlers import commands_router, listings_router
from services.database import Database
from utils.logging import configure_logging


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    bot = Bot(
        token=settings.bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher()
    dispatcher.include_router(commands_router)
    dispatcher.include_router(listings_router)
    database = Database(
        settings.database_url.get_secret_value() if settings.database_url else None
    )

    logger.info("Бот запущен")
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        await dispatcher.start_polling(bot, settings=settings, database=database)
    finally:
        await database.close()
        await bot.session.close()
        logger.info("Бот остановлен")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
