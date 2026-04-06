import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect(user='postgres', password='Affankhan0966', database='postgres', host='localhost', port=5432)
    try:
        ver = await conn.fetchval('SELECT version();')
        print("PG_VERSION_OUTPUT:", ver)
    finally:
        await conn.close()

if __name__ == '__main__':
    asyncio.run(main())
