from __future__ import annotations

import uuid

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.types import CHAR, TypeDecorator


class GUID(TypeDecorator[uuid.UUID]):
    """Platform-independent UUID type.

    PostgreSQL gets a native UUID column. SQLite and other backends store compact
    32-character UUID hex strings, which keeps unit tests dependency-light.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return value.hex

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))

