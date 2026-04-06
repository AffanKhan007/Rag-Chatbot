import asyncio
import asyncpg

async def setup():
    try:
        # connect to the default 'postgres' database
        conn = await asyncpg.connect(user='postgres', password='Affankhan0966', database='postgres', host='localhost', port=5433)
        try:
            await conn.execute("CREATE DATABASE rag_chatbot;")
            print("Database rag_chatbot created!")
        except asyncpg.exceptions.DuplicateDatabaseError:
            print("Database already exists.")
        finally:
            await conn.close()
        
        # connect to the new database to create the extension
        conn2 = await asyncpg.connect(user='postgres', password='Affankhan0966', database='rag_chatbot', host='localhost', port=5433)
        try:
            await conn2.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("Extension vector created!")
        finally:
            await conn2.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    asyncio.run(setup())
