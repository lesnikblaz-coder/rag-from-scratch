from sqlalchemy import Text
from sqlalchemy import UUID as sqlUUID
from sqlalchemy.orm import Mapped, mapped_column
from uuid import UUID, uuid4
from pgvector.sqlalchemy import VECTOR

from src.db.base import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(sqlUUID(as_uuid=True), primary_key=True, default=uuid4)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(768), nullable=False)