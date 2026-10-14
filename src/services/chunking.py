def chunk_text(text: str, *, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    if not text:
        return []

    if chunk_size < 10:
        raise ValueError("Invalid chunk size.")

    if overlap <= 0 or overlap > chunk_size: # 0 >= overlap > chunk_size
        raise ValueError("Invalid overlap value.")

    chunks = []
    step = chunk_size - overlap
    words = text.split()

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))

    return chunks