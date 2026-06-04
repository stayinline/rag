"""Create a default admin user for initial login."""
import asyncio
import json
import sys
import uuid

sys.path.insert(0, ".")

from app.auth import hash_password
from app.database import async_session
from app.models.tenant import Tenant, User

DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"
DEFAULT_EMAIL = "admin@example.com"


async def main():
    async with async_session() as session:
        # Check if user already exists
        from sqlalchemy import select

        result = await session.execute(select(User).where(User.username == DEFAULT_USERNAME))
        existing = result.scalar_one_or_none()
        if existing:
            print(f"User '{DEFAULT_USERNAME}' already exists, skipping.")
            return

        # Create default tenant
        tenant = Tenant(
            id=uuid.uuid4(),
            name="Default Org",
            description="Default organization",
            is_active=True,
        )
        session.add(tenant)
        await session.flush()

        # Create default admin user
        user = User(
            id=uuid.uuid4(),
            org_id=tenant.id,
            email=DEFAULT_EMAIL,
            username=DEFAULT_USERNAME,
            hashed_password=hash_password(DEFAULT_PASSWORD),
            is_active=True,
            roles=json.dumps(["admin"]),
        )
        session.add(user)
        await session.commit()
        print(f"Default user created: {DEFAULT_USERNAME} / {DEFAULT_PASSWORD}")


if __name__ == "__main__":
    asyncio.run(main())
