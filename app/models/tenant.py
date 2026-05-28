from sqlalchemy import Column, String, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base, TimestampMixin, UUIDMixin


class Tenant(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "tenants"

    name = Column(String(200), nullable=False)
    description = Column(Text, default="")
    is_active = Column(Boolean, default=True)


class Role(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "roles"

    org_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(String(100), nullable=False)
    permissions = Column(Text, default="[]")


class User(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "users"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    roles = Column(Text, default="[]")
