import asyncio
import sys

from aiogram import Bot

from config import get_settings


async def main(base_url: str) -> None:
    settings = get_settings()
    if settings.webhook_secret is None:
        raise RuntimeError("WEBHOOK_SECRET не задан")
    webhook_url = f"{base_url.rstrip('/')}/telegram/webhook"
    async with Bot(settings.bot_token.get_secret_value()) as bot:
        result = await bot.set_webhook(
            webhook_url,
            secret_token=settings.webhook_secret.get_secret_value(),
            allowed_updates=["message"],
            drop_pending_updates=False,
        )
    if not result:
        raise RuntimeError("Telegram не подтвердил установку webhook")
    print(f"Webhook установлен: {webhook_url}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Использование: python -m scripts.set_webhook https://project.vercel.app")
    asyncio.run(main(sys.argv[1]))
