# database_test.py
from sqlalchemy.ext.asyncio import create_async_engine
from app.config import DATABASE_URL
import asyncio

engine = create_async_engine(DATABASE_URL, echo=True)

async def test():
    conn = await engine.connect()
    print("Connected!")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(test())
