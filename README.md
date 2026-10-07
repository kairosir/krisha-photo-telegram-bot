# Telegram-бот для фотографий Krisha.kz

Асинхронный Telegram-бот на Python 3.12 и aiogram 3. Пользователь присылает ссылку на объявление Krisha.kz, бот находит доступные фотографии несколькими способами, скачивает их с ограничениями безопасности, обрабатывает и отправляет медиагруппами. Оригиналы по умолчанию дополнительно отправляются как документы без Telegram-сжатия.

Бот не обходит CAPTCHA, авторизацию и антибот-защиту. Если Krisha.kz ограничивает обычный HTTP-доступ, пользователь получает понятное сообщение об ошибке.

## Структура

```text
.
├── main.py                     # точка входа и polling
├── app.py                      # FastAPI webhook для Vercel
├── api/index.py                # точка входа Vercel Python Function
├── config.py                   # типизированные настройки из .env
├── handlers/
│   ├── commands.py             # /start, /help, /cancel
│   └── listings.py             # основной пользовательский сценарий
├── services/
│   ├── krisha.py               # валидация URL, парсинг и загрузка
│   └── image_processor.py      # сохранение оригинала и inpainting
├── utils/
│   ├── files.py                # гарантированная очистка temp-каталогов
│   ├── logging.py              # логирование
│   └── telegram.py             # медиагруппы по 10 файлов
├── tests/
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── vercel.json
```

## Как работает парсер

`services/krisha.py` изолирован от Telegram-логики. Он ищет изображения в:

1. JSON-LD;
2. JSON-состоянии frontend (`__NEXT_DATA__`, `__NUXT_DATA__` и JSON scripts);
3. URL внутри скриптов;
4. OpenGraph, preload, `srcset`, lazy-load атрибутах и обычных `<img>` как fallback.

Затем URL нормализуются, дубли URL удаляются, а после загрузки выполняется повторная дедупликация по SHA-256. Реальные миниатюры отсеиваются по разрешению. Для HTTP есть таймауты, повторы временных ошибок, проверка статуса, `Content-Type`, `Content-Length` и фактически прочитанного размера.

## Локальный запуск

Нужен Python 3.12+ и токен, полученный у [@BotFather](https://t.me/BotFather).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Откройте `.env`, замените `BOT_TOKEN` и запустите:

```bash
python main.py
```

Тесты:

```bash
python -m unittest discover -s tests -v
```

## Docker / VPS

Создайте `.env` из примера, затем:

```bash
docker compose up -d --build
docker compose logs -f bot
```

Остановка:

```bash
docker compose down
```

Для VPS достаточно установленного Docker Engine с Compose plugin. Контейнер работает от непривилегированного пользователя и автоматически перезапускается после сбоя или перезагрузки сервера.

## Vercel

На Vercel бот работает через Telegram webhook, а не polling. Добавьте в окружения Production, Preview и Development две секретные переменные:

- `BOT_TOKEN` — токен из BotFather;
- `WEBHOOK_SECRET` — случайная строка из латинских букв, цифр, `_` и `-`.

После production-деплоя зарегистрируйте webhook:

```bash
python -m scripts.set_webhook https://your-project.vercel.app
```

Корневой URL `/` служит health-check. Webhook доступен на `/telegram/webhook` и проверяет заголовок `X-Telegram-Bot-Api-Secret-Token`.

Для автоматической регистрации без передачи `BOT_TOKEN` наружу предусмотрен защищённый `POST /admin/setup-webhook`. Он принимает `WEBHOOK_SECRET` в заголовке `X-Setup-Secret`. Состояние можно проверить через защищённый `GET /admin/webhook-info`.

## Настройки `.env`

| Переменная | Значение по умолчанию | Назначение |
|---|---:|---|
| `BOT_TOKEN` | обязательно | токен Telegram-бота |
| `WEBHOOK_SECRET` | обязательно для Vercel | секрет проверки Telegram webhook |
| `LOG_LEVEL` | `INFO` | уровень логирования |
| `HTTP_TIMEOUT_SECONDS` | `25` | полный таймаут HTTP-запроса |
| `HTTP_RETRIES` | `3` | число попыток временно неуспешного запроса |
| `MAX_PAGE_BYTES` | `5242880` | максимум байт HTML-страницы |
| `MAX_IMAGE_BYTES` | `26214400` | максимум байт на изображение |
| `MAX_IMAGES_PER_LISTING` | `50` | максимум изображений объявления |
| `MIN_IMAGE_WIDTH` | `320` | минимальная ширина, фильтр миниатюр |
| `MIN_IMAGE_HEIGHT` | `240` | минимальная высота, фильтр миниатюр |
| `GLOBAL_PROCESSING_LIMIT` | `4` | глобальное число параллельных обработок |
| `DOWNLOAD_CONCURRENCY` | `5` | параллельные загрузки одной задачи |
| `SEND_ORIGINALS_AS_FILES` | `true` | отправлять исходники документами |

## Обработка и inpainting

Без маски `ImageProcessor.process()` копирует файл байт-в-байт: лишнего перекодирования нет. Для восстановления областей передайте прямоугольники или чёрно-белую маску:

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

Можно передать `mask_path=Path("mask.png")`: все ненулевые пиксели маски будут восстановлены алгоритмом OpenCV Telea. Этот режим следует использовать только для изображений, на обработку которых есть необходимые права.

Текущий Telegram-сценарий сохраняет изображения без изменений. Модуль и API inpainting готовы для подключения отдельного шага выбора маски или координат, не смешивая эту логику с парсером и обработчиками Telegram.

## Эксплуатационные замечания

- Telegram принимает не более 10 элементов в медиагруппе, поэтому длинные наборы автоматически разбиваются.
- Если Telegram отклоняет фото из-за формата, размера или геометрии, результат автоматически отправляется документом.
- Telegram может сжимать отправленные как фото изображения. Документы сохраняют исходные байты.
- На пользователя допускается одна активная задача. `/cancel` отменяет её, а временный каталог удаляется в `finally` контекстного менеджера.
- При изменении верстки Krisha.kz достаточно обновить `services/krisha.py` и его тесты.
- Соблюдайте условия использования Krisha.kz и права владельцев изображений.
