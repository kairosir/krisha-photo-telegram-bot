import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import Update
from fastapi import FastAPI, HTTPException, Request

from bot.handlers import router
from config import get_settings
from services.database import Database
from utils.logger import configure_logging

logger = logging.getLogger(__name__)
_bot: Bot | None = None
_dispatcher: Dispatcher | None = None
_database: Database | None = None


def _components() -> tuple[Bot, Dispatcher]:
    global _bot, _dispatcher
    if _bot is None or _dispatcher is None:
        settings = get_settings()
        configure_logging(settings.log_level)
        _bot = Bot(
            token=settings.bot_token.get_secret_value(),
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        _dispatcher = Dispatcher()
        _dispatcher.include_router(router)
    return _bot, _dispatcher


def _get_database() -> Database:
    global _database
    if _database is None:
        settings = get_settings()
        _database = Database(
            settings.database_url.get_secret_value() if settings.database_url else None
        )
    return _database


async def _dispatch_update(bot: Bot, dispatcher: Dispatcher, update: Update) -> None:
    await dispatcher.feed_update(
        bot,
        update,
        settings=get_settings(),
        database=_get_database(),
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    if _bot is not None:
        await _bot.session.close()
    if _database is not None:
        await _database.close()


app = FastAPI(title="Krisha Telegram Bot", lifespan=lifespan)


def _verify_secret(request: Request, header_name: str) -> str:
    settings = get_settings()
    expected = settings.webhook_secret.get_secret_value() if settings.webhook_secret else ""
    received = request.headers.get(header_name, "")
    if not expected or not hmac.compare_digest(received, expected):
        raise HTTPException(status_code=403, detail="Invalid secret")
    return expected


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "krisha-telegram-bot"}


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request) -> dict[str, bool]:
    _verify_secret(request, "X-Telegram-Bot-Api-Secret-Token")
    try:
        payload = await request.json()
        bot, dispatcher = _components()
        update = Update.model_validate(payload, context={"bot": bot})
        await _dispatch_update(bot, dispatcher, update)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Ошибка обработки Telegram webhook")
        raise HTTPException(status_code=500, detail="Webhook processing failed")
    return {"ok": True}


@app.post("/admin/setup-webhook")
async def setup_webhook(request: Request) -> dict[str, str | bool]:
    secret = _verify_secret(request, "X-Setup-Secret")
    bot, _ = _components()
    webhook_url = f"{str(request.base_url).rstrip('/')}/telegram/webhook"
    result = await bot.set_webhook(
        webhook_url,
        secret_token=secret,
        allowed_updates=["message"],
        drop_pending_updates=False,
    )
    return {"ok": result, "webhook_url": webhook_url}


@app.get("/admin/webhook-info")
async def webhook_info(request: Request) -> dict[str, object]:
    _verify_secret(request, "X-Setup-Secret")
    bot, _ = _components()
    info = await bot.get_webhook_info()
    return {
        "url": info.url,
        "pending_update_count": info.pending_update_count,
        "last_error_date": info.last_error_date.isoformat() if info.last_error_date else None,
        "last_error_message": info.last_error_message,
    }


async def run_polling() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    bot, dispatcher = _components()
    database = _get_database()
    logger.info("Бот запущен")
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        await dispatcher.start_polling(bot, settings=settings, database=database)
    finally:
        await database.close()
        await bot.session.close()
        logger.info("Бот остановлен")


def main() -> None:
    try:
        asyncio.run(run_polling())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
