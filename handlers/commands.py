from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from handlers.listings import cancel_user_task

router = Router(name="commands")


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.answer(
        "Привет! 👋\n"
        "Отправь мне ссылку на объявление Krisha.kz, и я получу доступные "
        "фотографии объявления для обработки."
    )


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    await message.answer(
        "Пришли ссылку вида https://krisha.kz/a/show/12345678.\n\n"
        "Я найду доступные фотографии, сохраню их качество и отправлю результат. "
        "Оригиналы также могут быть отправлены как файлы.\n\n"
        "Команды:\n"
        "/start — начать работу\n"
        "/help — эта справка\n"
        "/cancel — отменить текущую задачу\n\n"
        "Используй обработку только для изображений, на которые у тебя есть необходимые права."
    )


@router.message(Command("cancel"))
async def cancel_handler(message: Message) -> None:
    if message.from_user and cancel_user_task(message.from_user.id):
        await message.answer("🛑 Текущая задача отменена.")
    else:
        await message.answer("Сейчас нет активной задачи.")
