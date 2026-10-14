import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

import asyncpg

logger = logging.getLogger(__name__)


class Database:
    """Опциональный журнал задач. Ошибки БД не прерывают работу бота."""

    def __init__(self, dsn: str | None) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._dsn)

    async def _get_pool(self) -> asyncpg.Pool | None:
        if not self._dsn:
            return None
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=0,
                max_size=3,
                command_timeout=10,
                max_inactive_connection_lifetime=60,
            )
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def create_job(self, user_id: int, listing_url: str) -> UUID | None:
        job_id = uuid4()
        pool = await self._safe_pool()
        if pool is None:
            return None
        try:
            await pool.execute(
                """
                INSERT INTO bot_jobs (id, telegram_user_id, listing_url, status)
                VALUES ($1, $2, $3, 'processing')
                """,
                job_id,
                user_id,
                listing_url,
            )
            return job_id
        except Exception:
            logger.exception("Не удалось записать начало задачи в Neon")
            return None

    async def complete_job(self, job_id: UUID | None, image_count: int) -> None:
        await self._finish_job(job_id, "completed", image_count=image_count)

    async def fail_job(self, job_id: UUID | None, error: str) -> None:
        await self._finish_job(job_id, "failed", error=error[:1000])

    async def cancel_job(self, job_id: UUID | None) -> None:
        await self._finish_job(job_id, "cancelled")

    async def _finish_job(
        self,
        job_id: UUID | None,
        status: str,
        *,
        image_count: int | None = None,
        error: str | None = None,
    ) -> None:
        if job_id is None:
            return
        pool = await self._safe_pool()
        if pool is None:
            return
        try:
            await pool.execute(
                """
                UPDATE bot_jobs
                SET status = $2,
                    image_count = COALESCE($3, image_count),
                    error_message = $4,
                    finished_at = $5,
                    updated_at = $5
                WHERE id = $1
                """,
                job_id,
                status,
                image_count,
                error,
                datetime.now(timezone.utc),
            )
        except Exception:
            logger.exception("Не удалось обновить задачу в Neon")

    async def _safe_pool(self) -> asyncpg.Pool | None:
        try:
            return await self._get_pool()
        except Exception:
            logger.exception("Не удалось подключиться к Neon")
            return None
