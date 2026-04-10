# modules/sql_module.py (ASYNC VERSION)
import asyncpg
import time

class SQLModule:
    def __init__(self, config):
        self.config = config
        self.pool = None

    async def init(self):
        self.pool = await asyncpg.create_pool(
            database=self.config["dbname"],
            user=self.config["user"],
            password=self.config["password"],
            host=self.config["host"],
            port=self.config["port"],
            min_size=1,
            max_size=5
        )
        await self._init_tables()

    async def _init_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id SERIAL PRIMARY KEY,
                    text TEXT,
                    tags TEXT,
                    timestamp DOUBLE PRECISION
                );
            """)

    async def store(self, text, tags):
        ts = time.time()
        tag_str = ",".join(tags)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO memories (text, tags, timestamp) VALUES ($1, $2, $3) RETURNING id",
                text, tag_str, ts
            )

        return {"status": "ok", "id": row["id"]}

    async def query(self, query_text):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, text, tags, timestamp FROM memories WHERE text ILIKE $1",
                f"%{query_text}%"
            )

        return {"results": [dict(r) for r in rows]}