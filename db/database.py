"""
db/database.py  (SQLite edition)
---------------------------------
Uses aiosqlite + SQLAlchemy 2 async so the app runs without Docker/Postgres.
The DB file is created at ./governance.db automatically on first startup.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)

_engine = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None
DB_FILE = Path(__file__).parent.parent / "governance.db"


def setup_engine(database_url: str) -> None:
    global _engine, _async_session_factory
    # Always use SQLite for local dev — ignore postgres url from env
    sqlite_url = f"sqlite+aiosqlite:///{DB_FILE}"
    _engine = create_async_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})
    _async_session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    logger.info(f"Database engine initialized (SQLite: {DB_FILE})")


async def init_db() -> None:
    if _engine is None:
        raise RuntimeError("Call setup_engine() before init_db().")

    # SQLite-compatible DDL (no gen_random_uuid, no NUMERIC, no pg types)
    ddl = """
    CREATE TABLE IF NOT EXISTS governance_logs (
        request_id          TEXT PRIMARY KEY,
        product_id          TEXT NOT NULL,
        bot_id              TEXT NOT NULL,
        bot_reasoning       TEXT,
        original_price      REAL NOT NULL,
        proposed_price      REAL NOT NULL,
        price_delta_pct     REAL NOT NULL,
        status              TEXT NOT NULL DEFAULT 'pending',
        layer1_result       TEXT,
        ai_reasoning        TEXT,
        ai_confidence_score REAL,
        human_reviewer_id   TEXT,
        human_decision      TEXT,
        created_at          TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_status ON governance_logs (status);
    CREATE INDEX IF NOT EXISTS idx_product ON governance_logs (product_id, created_at DESC);
    """

    async with _engine.begin() as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                await conn.execute(text(s))
    logger.info("SQLite schema ready.")


async def close_db() -> None:
    if _engine:
        await _engine.dispose()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if _async_session_factory is None:
        raise RuntimeError("Call setup_engine() first.")
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with get_db_session() as session:
        yield session
