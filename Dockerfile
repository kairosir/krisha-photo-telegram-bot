FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Имена пакетов различаются между Debian bookworm (старые имена) и trixie (t64-суффикс),
# поэтому ставим все известные варианты; отсутствующие в данном релизе пакеты игнорируются.
RUN apt-get update \
    && for pkg in libglib2.0-0 libgl1 libdbus-glib-1-2 curl ca-certificates \
       libgtk-3-0 libgtk-3-0t64 \
       libdbus-glib-1-0 libdbus-glib-1-0t64 \
       libasound2 libasound2t64; do \
         apt-get install -y --no-install-recommends "$pkg" 2>/dev/null || true; \
       done \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && python -m playwright install firefox \
    && python -m camoufox fetch

COPY . .

# Модель LaMa (~200 МБ) кладём прямо в образ, чтобы не качать при первом запросе.
# Если скачивание недоступно — бот сам докачает модель в рантайме (fallback в коде).
RUN mkdir -p /app/models \
    && (curl -L --fail --retry 3 -o /app/models/big-lama.pt \
        https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt \
        || echo "WARN: LaMa weights not baked, will download at runtime")

RUN useradd --create-home --uid 10001 botuser && chown -R botuser:botuser /app \
    && mkdir -p /data/camoufox && chown -R botuser:botuser /data
USER botuser

ENV LAMA_MODEL_DIR=/app/models

CMD ["python", "-m", "bot.main"]
