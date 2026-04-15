import asyncio

from app.main import initialize_database


async def main():
    await initialize_database()
    print(
        "Database schema initialized. Start the API, create a workspace, and upload files through the new UI."
    )


if __name__ == "__main__":
    asyncio.run(main())
