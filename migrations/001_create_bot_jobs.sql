CREATE TABLE IF NOT EXISTS bot_jobs (
    id UUID PRIMARY KEY,
    telegram_user_id BIGINT NOT NULL,
    listing_url TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')
    ),
    image_count INTEGER NOT NULL DEFAULT 0 CHECK (image_count >= 0),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS bot_jobs_user_created_idx
    ON bot_jobs (telegram_user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS bot_jobs_status_created_idx
    ON bot_jobs (status, created_at DESC);
