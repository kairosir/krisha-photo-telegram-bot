from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def commands_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="/help"), KeyboardButton(text="/cancel")]],
        resize_keyboard=True,
        input_field_placeholder="Ссылка на объявление Krisha.kz",
    )
