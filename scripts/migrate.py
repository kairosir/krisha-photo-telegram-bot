import asyncio
import os
from pathlib import Path

import asyncpg


async def main() -> None:
    database_url = os.environ.get("DATABASE_URL_UNPOOLED")
    if not database_url:
        raise RuntimeError("DATABASE_URL_UNPOOLED не задан")

    migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
    connection = await asyncpg.connect(database_url, command_timeout=30)
    try:
        for migration in sorted(migrations_dir.glob("*.sql")):
            await connection.execute(migration.read_text(encoding="utf-8"))
            print(f"Применена миграция: {migration.name}")
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
