import asyncio
from sqlalchemy import text
from app.db import SessionLocal, engine
from app.models import Base, Knowledge
from app.rag import get_embedding

data = [
    ("Newton First Law", "An object in motion stays in motion unless acted upon by a force."),
    ("Newton Second Law", "Force equals mass times acceleration."),
    ("Newton Third Law", "Every action has equal and opposite reaction."),
    ("Gravity", "Gravity is a force that attracts objects toward each other."),
    ("Energy", "Energy cannot be created or destroyed, only transformed.")
]

async def main():
    # create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Create an HNSW index for fast approximate cosine-distance search.
        # Idempotent so you can re-run `seed.py` safely.
        await conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS knowledge_embedding_hnsw_idx
                ON knowledge
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
        )

    # insert data
    async with SessionLocal() as db:
        for topic, content in data:
            emb = await get_embedding(content)
            db.add(Knowledge(topic=topic, content=content, embedding=emb))
        await db.commit()

    print("Data seeded successfully!")

if __name__ == "__main__":
    asyncio.run(main())