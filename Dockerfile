FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
       libgtk-3-0 libdbus-glib-1-0 libasound2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && python -m playwright install firefox \
    && python -m camoufox fetch

COPY . .

RUN useradd --create-home --uid 10001 botuser && chown -R botuser:botuser /app \
    && mkdir -p /data/camoufox && chown -R botuser:botuser /data
USER botuser

CMD ["python", "-m", "bot.main"]
