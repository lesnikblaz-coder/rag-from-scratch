from sqlalchemy import Text, String, Integer, UniqueConstraint
from sqlalchemy import UUID as sqlUUID
from sqlalchemy.orm import Mapped, mapped_column
from uuid import UUID, uuid4
from pgvector.sqlalchemy import VECTOR

from src.db.base import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    __table_args__ = (
        UniqueConstraint(
            "source_file",
            "chunk_index",
            name="uq_source_chunk"
        ),
    )

    id: Mapped[UUID] = mapped_column(sqlUUID(as_uuid=True), primary_key=True, default=uuid4)
    source_file: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(768), nullable=False)