import hmac
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import Update
from fastapi import FastAPI, HTTPException, Request

from config import get_settings
from handlers import commands_router, listings_router
from utils.logging import configure_logging

logger = logging.getLogger(__name__)
_bot: Bot | None = None
_dispatcher: Dispatcher | None = None


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
        _dispatcher.include_router(commands_router)
        _dispatcher.include_router(listings_router)
    return _bot, _dispatcher


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    if _bot is not None:
        await _bot.session.close()


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
        await dispatcher.feed_update(bot, update)
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
