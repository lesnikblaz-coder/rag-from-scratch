from pathlib import Path

from src.services.document_loader import load_text_file
from src.services.chunking import chunk_text

sample_dir = Path(__file__).resolve().parent.parent / "sample.txt"
text = load_text_file(sample_dir)

chunks = chunk_text(
    text,
    chunk_size=300,
    overlap=40
)

print("DOCUMENT:", sample_dir.name)
print("Total words: ", len(text.split()))
print("Total chunks: ", len(chunks))

for i, c in enumerate(chunks, start=1):
    print("\n================================")
    print(f"CHUNK {i}")
    print(f"Words: {len(c.split())}")
    print("================================\n")
    print(c)