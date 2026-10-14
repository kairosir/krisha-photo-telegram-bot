import unittest
from unittest.mock import AsyncMock, patch

from bot.main import _dispatch_update


class WebhookDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_passes_settings_to_aiogram_dispatcher(self) -> None:
        bot = object()
        update = object()
        dispatcher = AsyncMock()
        settings = object()
        database = object()

        with (
            patch("bot.main.get_settings", return_value=settings),
            patch("bot.main._get_database", return_value=database),
        ):
            await _dispatch_update(bot, dispatcher, update)  # type: ignore[arg-type]

        dispatcher.feed_update.assert_awaited_once_with(
            bot,
            update,
            settings=settings,
            database=database,
        )


if __name__ == "__main__":
    unittest.main()
