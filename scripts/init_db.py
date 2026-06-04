"""Run initial database migration."""
import asyncio
import sys

sys.path.insert(0, ".")

from app.models.base import Base
# Import all models so they are registered with Base.metadata
import app.models.tenant  # noqa: F401
import app.models.kb  # noqa: F401
import app.models.document  # noqa: F401
import app.models.audit  # noqa: F401
import app.models.chunk  # noqa: F401
import app.models.conversation  # noqa: F401
import app.models.task  # noqa: F401
import app.models.paper  # noqa: F401
from app.database import engine


async def main():
    print("Running database migration...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Database tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
