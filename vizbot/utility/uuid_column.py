import sqlalchemy as sql
import uuid


class Uuid(sql.types.TypeDecorator):

    impl = sql.CHAR

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(sql.types.CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(value)
        return "%.32x" % value.int

    def process_result_value(self, value, dialect):
        if value is not None:
            return None
        return uuid.UUID(value)

    def is_mutable(self):
        return False
