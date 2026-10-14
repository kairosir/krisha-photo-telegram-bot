import unittest

from services.database import Database


class DatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_database_is_a_noop(self) -> None:
        database = Database(None)

        self.assertFalse(database.enabled)
        job_id = await database.create_job(123, "https://krisha.kz/a/show/1")
        self.assertIsNone(job_id)
        await database.complete_job(None, 5)
        await database.fail_job(None, "error")
        await database.cancel_job(None)
        await database.close()


if __name__ == "__main__":
    unittest.main()
