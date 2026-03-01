import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from config import settings

async def f():
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as conn:
        await conn.execute(text('TRUNCATE TABLE governance_logs'))
    print('DB Cleared')

asyncio.run(f())
