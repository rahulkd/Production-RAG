# Find all chunks for a given arxiv_id and write their text to ./results/.
from opensearchpy import OpenSearch

import os as _os, sys as _sys
# Ensure the repo root is importable so `src.*` resolves when run directly.
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "pyproject.toml")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

# Top-level (hybrid) client -> targets the chunk index: {index_name}-{chunk_index_suffix}
from src.services.opensearch.factory import make_opensearch_client

# ============================ CONFIGURATION ============================
HOST = "http://localhost:9200"

# Index to read from. Set to a specific name, or leave as None to use the
# client's default hybrid chunk index ({index_name}-{chunk_index_suffix}).
#INDEX_NAME = "arxiv-papers-chunks"
INDEX_NAME = "arxiv_evaluation_index"

# The paper to dump chunks for.
ARXIV_ID = "2604.05432v1"
# ======================================================================

RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results")

print("PAPER CHUNKS DUMP")
print("=" * 40)

# Create OpenSearch client using factory pattern
opensearch_client = make_opensearch_client()

# Override for local execution (localhost instead of container hostname)
opensearch_client.host = HOST
opensearch_client.client = OpenSearch(
    hosts=[HOST],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

# Override the target index if one is configured above.
if INDEX_NAME:
    opensearch_client.index_name = INDEX_NAME

print(f"Index:   {opensearch_client.index_name}")
print(f"arxiv_id: {ARXIV_ID}\n")

# Fetch all chunks for the paper, sorted by chunk_index (embedding excluded).
chunks = opensearch_client.get_chunks_by_paper(ARXIV_ID)

if not chunks:
    print(f"⚠️  No chunks found for arxiv_id '{ARXIV_ID}' in '{opensearch_client.index_name}'.")
    raise SystemExit(0)

title = chunks[0].get("title") or "Unknown"
print(f"✓ Found {len(chunks)} chunks for: {title[:60]}")

# Write the chunk text to results/<arxiv_id>_chunks.txt
_os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = _os.path.join(RESULTS_DIR, f"{ARXIV_ID}_chunks.txt")

with open(out_path, "w") as fh:
    fh.write(f"arxiv_id: {ARXIV_ID}\n")
    fh.write(f"title: {title}\n")
    fh.write(f"index: {opensearch_client.index_name}\n")
    fh.write(f"total chunks: {len(chunks)}\n")
    fh.write("=" * 70 + "\n\n")

    for chunk in chunks:
        idx = chunk.get("chunk_index", "?")
        section = chunk.get("section_title") or "(no section)"
        words = chunk.get("chunk_word_count", "?")
        text = chunk.get("chunk_text") or ""

        fh.write(f"----- chunk #{idx} | section: {section} | words: {words} -----\n")
        fh.write(text.strip() + "\n\n")

print(f"\n✓ Wrote {len(chunks)} chunks to: {_os.path.relpath(out_path, _root)}")
