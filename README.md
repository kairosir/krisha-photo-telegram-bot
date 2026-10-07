# Photo Processing Bot

Асинхронный Telegram-бот на Python 3.12 и aiogram 3. Пользователь отправляет ссылку на объявление Krisha.kz, бот находит доступные фотографии несколькими способами, безопасно скачивает их, исключает миниатюры и дубликаты и возвращает медиагруппами. Исходники можно дополнительно отправлять документами без Telegram-сжатия.

Бот не обходит CAPTCHA, авторизацию или антибот-защиту. Если страница недоступна обычным HTTP-запросом, пользователь получает понятное сообщение.

## Структура

```text
photo-processing-bot/
├── bot/
│   ├── __init__.py
│   ├── main.py                 # polling и FastAPI webhook
│   ├── handlers.py             # команды и обработка ссылок
│   ├── keyboards.py
│   └── messages.py
├── services/
│   ├── __init__.py
│   ├── source.py               # получение страницы и поиск фотографий
│   ├── downloader.py           # безопасная параллельная загрузка
│   ├── image_processor.py      # копирование и inpainting по явной маске
│   ├── uploader.py             # отправка в Telegram
│   └── database.py             # опциональный журнал задач в Neon
├── utils/
│   ├── __init__.py
│   ├── validators.py
│   ├── files.py
│   └── logger.py
├── temp/
│   ├── input/
│   └── output/
├── config/
│   ├── __init__.py
│   └── settings.py
├── api/index.py                # точка входа Vercel
├── migrations/                 # SQL-миграции Neon
├── scripts/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── vercel.json
```

Дополнительные каталоги `api`, `migrations`, `scripts` и `tests` нужны для Vercel, Neon и проверки проекта. Настоящий `.env` исключён из Git.

## Получение фотографий

Логика Krisha.kz изолирована в `services/source.py`. Источники проверяются независимо:

1. JSON-LD;
2. frontend JSON (`__NEXT_DATA__`, `__NUXT_DATA__` и другие JSON-скрипты);
3. URL в JavaScript;
4. OpenGraph, preload, `srcset`, lazy-load атрибуты и обычные `<img>`.

URL нормализуются и дедуплицируются. После загрузки файлы повторно дедуплицируются по SHA-256, а миниатюры отбрасываются по фактическому разрешению. HTTP-клиент использует User-Agent, таймауты, retry, проверку статуса, `Content-Type`, `Content-Length` и фактически загруженного объёма.

## Локальный запуск

Нужны Python 3.12+ и токен от [@BotFather](https://t.me/BotFather).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Укажите `BOT_TOKEN` в `.env` и запустите:

```bash
python -m bot.main
```

Тесты:

```bash
python -m unittest discover -s tests -v
```

## Docker / VPS

```bash
docker compose up -d --build
docker compose logs -f bot
```

Остановка: `docker compose down`. Контейнер работает от непривилегированного пользователя. Временный каталог каждой задачи удаляется после успеха, ошибки или отмены.

## Vercel

На Vercel бот работает через webhook. Добавьте переменные `BOT_TOKEN`, `WEBHOOK_SECRET` и, при необходимости, pooled `DATABASE_URL` Neon. После production-деплоя зарегистрируйте webhook:

```bash
python -m scripts.set_webhook https://your-project.vercel.app
```

Корневой URL `/` — health-check. `/telegram/webhook` проверяет заголовок `X-Telegram-Bot-Api-Secret-Token`. Защищённые `/admin/setup-webhook` и `/admin/webhook-info` используют `WEBHOOK_SECRET` в `X-Setup-Secret`.

## Neon

База необязательна: без `DATABASE_URL` бот продолжит работать. Для применения схемы:

```bash
python -m scripts.migrate
```

В базе хранятся только пользователь, URL, статус задачи, число файлов, ошибка и временные метки. Фотографии в Neon не записываются.

## Настройки `.env`

| Переменная | По умолчанию | Назначение |
|---|---:|---|
| `BOT_TOKEN` | обязательно | токен Telegram-бота |
| `WEBHOOK_SECRET` | для Vercel | секрет webhook |
| `DATABASE_URL` | пусто | pooled Neon URL |
| `LOG_LEVEL` | `INFO` | уровень логирования |
| `HTTP_TIMEOUT_SECONDS` | `25` | таймаут HTTP-запроса |
| `HTTP_RETRIES` | `3` | число попыток |
| `MAX_PAGE_BYTES` | `5242880` | предел размера страницы |
| `MAX_IMAGE_BYTES` | `26214400` | предел одного изображения |
| `MAX_IMAGES_PER_LISTING` | `50` | максимум фотографий |
| `MIN_IMAGE_WIDTH` | `320` | минимальная ширина |
| `MIN_IMAGE_HEIGHT` | `240` | минимальная высота |
| `GLOBAL_PROCESSING_LIMIT` | `4` | глобальный лимит обработки |
| `DOWNLOAD_CONCURRENCY` | `5` | параллельные загрузки |
| `SEND_ORIGINALS_AS_FILES` | `true` | отправлять исходники файлами |

## Обработка изображений

Без маски `ImageProcessor.process()` копирует файл байт-в-байт. Для восстановления повреждённой области собственного исходника можно передать подготовленную маску или координаты:

```python
import asyncio
from pathlib import Path
from services.image_processor import ImageProcessor, InpaintRegion

processor = ImageProcessor(asyncio.Semaphore(4))
await processor.process(
    Path("source.jpg"),
    Path("result.jpg"),
    regions=[InpaintRegion(x=100, y=80, width=240, height=60)],
)
```

Ненулевые пиксели `mask_path` восстанавливаются OpenCV Telea. Автоматическое обнаружение или удаление водяных знаков в Telegram-сценарий не входит.

## Эксплуатация

- Медиагруппы разбиваются по 10 элементов.
- Неподходящее для фото изображение отправляется документом.
- Для пользователя выполняется одна задача; `/cancel` отменяет её.
- Глобальный semaphore ограничивает обработку.
- При изменении сайта достаточно обновить `services/source.py` и тесты.
