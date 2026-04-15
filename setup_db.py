import asyncio
from urllib.parse import urlparse

import asyncpg

from app.config import DATABASE_URL


def parse_database_settings():
    parsed = urlparse(DATABASE_URL)
    return {
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": (parsed.path or "").lstrip("/") or "rag_chatbot",
    }


async def setup():
    settings = parse_database_settings()

    admin_connection = await asyncpg.connect(
        user=settings["user"],
        password=settings["password"],
        database="postgres",
        host=settings["host"],
        port=settings["port"],
    )
    try:
        try:
            await admin_connection.execute(f"CREATE DATABASE {settings['database']};")
            print(f"Database {settings['database']} created.")
        except asyncpg.exceptions.DuplicateDatabaseError:
            print(f"Database {settings['database']} already exists.")
    finally:
        await admin_connection.close()

    database_connection = await asyncpg.connect(
        user=settings["user"],
        password=settings["password"],
        database=settings["database"],
        host=settings["host"],
        port=settings["port"],
    )
    try:
        await database_connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("Vector extension is enabled.")
    finally:
        await database_connection.close()


if __name__ == "__main__":
    asyncio.run(setup())
