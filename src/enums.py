from enum import StrEnum, IntEnum


class EmbeddingTaskTypes(StrEnum):
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"

class EmbeddingDimensions(IntEnum):
    OUTPUT_DIM = 768

class GroundingLabel(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"