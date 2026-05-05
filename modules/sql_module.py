from typing import Optional, cast
import asyncpg
import time
import logging
import json
import numpy as np
logger = logging.getLogger("SQLModule")

class SQLModule:
    def __init__(self, config):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None

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
        pool = cast(asyncpg.Pool, self.pool)
        async with pool.acquire() as conn:

            # Existing table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id SERIAL PRIMARY KEY,
                    text TEXT,
                    tags TEXT,
                    timestamp DOUBLE PRECISION
                );
            """)

            # New semantic memory table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS hal_semantic_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # New episodic memory table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS hal_vector_memory (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    vector_json JSONB,
                    timestamp DOUBLE PRECISION
                );
            """)

        logger.info("[SQLModule] Tables ready")

    async def store(self, text, tags):
        ts = time.time()
        tag_str = ",".join(tags)
        logger.info(f"[SQLModule] Storing memory: '{text[:100]}' tags={tags}")
        pool = cast(asyncpg.Pool, self.pool)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO memories (text, tags, timestamp) VALUES ($1, $2, $3) RETURNING id",
                text, tag_str, ts
            )

        logger.info(f"[SQLModule] Stored with id={row['id']}")
        return {"status": "ok", "id": row["id"]}

    async def query(self, query_text):
        logger.info(f"[SQLModule] Querying memories for: '{query_text}'")
        pool = cast(asyncpg.Pool, self.pool)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, text, tags, timestamp FROM memories WHERE text ILIKE $1",
                f"%{query_text}%"
            )

        logger.info(f"[SQLModule] Query returned {len(rows)} rows")
        return {"results": [dict(r) for r in rows]}

    async def semantic_write(self, key: str, value: str):
        pool = cast(asyncpg.Pool, self.pool)
        async with pool.acquire() as conn:
            conn = cast(asyncpg.Connection, conn)

            await conn.execute("""
                INSERT INTO hal_semantic_memory (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();
            """, key, value)

        return {"status": "ok"}

    async def semantic_read(self, key: str):
        pool = cast(asyncpg.Pool, self.pool)
        async with pool.acquire() as conn:
            conn = cast(asyncpg.Connection, conn)

            row = await conn.fetchrow(
                "SELECT value FROM hal_semantic_memory WHERE key = $1",
                key
            )

        return {"value": row["value"] if row else None}


    async def vector_write(self, content: str, vector_json: dict, timestamp: float):
        pool = cast(asyncpg.Pool, self.pool)
        async with pool.acquire() as conn:
            conn = cast(asyncpg.Connection, conn)

            await conn.execute("""
                INSERT INTO hal_vector_memory (content, vector_json, timestamp)
                VALUES ($1, $2, $3);
            """, content, json.dumps(vector_json), timestamp)

        return {"status": "ok"}



    async def vector_search(self, query_vector: list[float], top_k: int = 5):
        pool = cast(asyncpg.Pool, self.pool)
        async with pool.acquire() as conn:
            conn = cast(asyncpg.Connection, conn)

            rows = await conn.fetch("""
                SELECT id, content, vector_json, timestamp
                FROM hal_vector_memory
            """)

        qv = np.array(query_vector, dtype=np.float32)
        scored = []

        for row in rows:
            vec = np.array(json.loads(row["vector_json"]), dtype=np.float32)
            score = float(np.dot(qv, vec) / (np.linalg.norm(qv) * np.linalg.norm(vec)))
            scored.append((score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return {"results": [item[1] for item in scored[:top_k]]}
