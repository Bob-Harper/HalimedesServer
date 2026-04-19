import asyncpg
import time
import logging

logger = logging.getLogger("SQLModule")

class SQLModule:
    def __init__(self, config):
        self.config = config
        self.pool = None

    async def init(self):
        logger.info("[SQLModule] Initializing connection pool")
        self.pool = await asyncpg.create_pool(
            database=self.config["dbname"],
            user=self.config["user"],
            password=self.config["password"],
            host=self.config["host"],
            port=self.config["port"],
            min_size=1,
            max_size=5
        )
        logger.info("[SQLModule] Pool created")
        await self._init_tables()

    async def _init_tables(self):
        logger.info("[SQLModule] Ensuring tables exist")
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id SERIAL PRIMARY KEY,
                    text TEXT,
                    tags TEXT,
                    timestamp DOUBLE PRECISION
                );
            """)
        logger.info("[SQLModule] Tables ready")

    async def store(self, text, tags):
        ts = time.time()
        tag_str = ",".join(tags)
        logger.info(f"[SQLModule] Storing memory: '{text[:100]}' tags={tags}")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO memories (text, tags, timestamp) VALUES ($1, $2, $3) RETURNING id",
                text, tag_str, ts
            )

        logger.info(f"[SQLModule] Stored with id={row['id']}")
        return {"status": "ok", "id": row["id"]}

    async def query(self, query_text):
        logger.info(f"[SQLModule] Querying memories for: '{query_text}'")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, text, tags, timestamp FROM memories WHERE text ILIKE $1",
                f"%{query_text}%"
            )

        logger.info(f"[SQLModule] Query returned {len(rows)} rows")
        return {"results": [dict(r) for r in rows]}