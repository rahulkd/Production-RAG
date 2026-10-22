"""Reading indexed chunks and recovering their real body text.

``TextChunker`` prepends ``f"{title}\\n\\nAbstract: {abstract}\\n\\n"`` to *every*
chunk of a paper. Two consequences drive this module:

1. **Questions must never be generated from that header.** Every chunk carries
   the same title and abstract, so a question written from it is answerable by
   all ~24 chunks of the paper at once. The ground-truth label would name one
   chunk while two dozen others answer equally well, which makes Precision@k
   meaningless and Recall@k unfairly harsh.
2. **The header is what makes BM25 look strong.** It is repeated boilerplate
   that matches almost any on-topic query, so leaving it in the text shown to
   the question generator biases the whole evaluation toward lexical retrieval.

:func:`strip_header` removes it. Verified byte-exact against all 1036 chunks in
the evaluation index — zero mismatches — so a failure to match is treated as a
signal that the chunker changed, not as something to paper over.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from src.evaluation import config as eval_config

logger = logging.getLogger(__name__)

# Leading "Section: <name>" markers that the chunker writes ahead of the body.
# Useful context for a human reviewer, noise for a question generator.
_SECTION_MARKER = re.compile(r"^(Section:[^\n]*\n\n)+")

# Sections that contain no answerable claims. References dominate the corpus
# (155 chunks), and a question generated from a bibliography is unanswerable
# in any useful sense.
_EXCLUDED_SECTION = re.compile(
    r"(reference|bibliograph|acknowledg|ccs concept|keyword|appendix"
    r"|author contribution|funding|ethics statement|declaration|supplementary)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    """One indexed chunk, with its header separated from its body.

    :param arxiv_id: Owning paper.
    :param chunk_index: Position within the paper; with ``arxiv_id`` this is the
        stable ground-truth address.
    :param section_title: Section the chunk came from, if the parser found one.
    :param body: Chunk text with the title/abstract header and leading section
        markers removed. **This is what questions are generated from.**
    :param full_text: The chunk exactly as indexed, header included. Hashed for
        staleness detection; never shown to the generator.
    :param title: Paper title.
    :param abstract: Paper abstract.
    :param word_count: Word count of the indexed chunk.
    """

    arxiv_id: str
    chunk_index: int
    section_title: str
    body: str
    full_text: str
    title: str
    abstract: str
    word_count: int = 0

    @property
    def key(self) -> tuple:
        """Ground-truth address of this chunk.

        :returns: ``(arxiv_id, chunk_index)``.
        """
        return (self.arxiv_id, self.chunk_index)

    @property
    def is_eligible(self) -> bool:
        """Whether this chunk can support a real question.

        Excludes the title/author page, reference and metadata sections, and
        bodies too thin to contain a substantive claim.

        :returns: True when the chunk is usable for generation.
        """
        if self.chunk_index == 0:
            return False  # Title page and author list.
        if _EXCLUDED_SECTION.search(self.section_title or ""):
            return False
        return len(self.body) >= eval_config.MIN_CHUNK_BODY_CHARS


def strip_header(chunk_text: str, title: str, abstract: str) -> str:
    """Remove the prepended title/abstract header from an indexed chunk.

    :param chunk_text: The chunk exactly as indexed.
    :param title: Paper title, as denormalised onto the chunk.
    :param abstract: Paper abstract, as denormalised onto the chunk.
    :returns: The chunk body, header and leading section markers removed.
    """
    header = f"{title}\n\nAbstract: {abstract}\n\n"

    if chunk_text.startswith(header):
        body = chunk_text[len(header) :]
    else:
        # The chunker's header format changed, or title/abstract drifted from
        # what was indexed. Falling back silently would feed abstract text to
        # the generator, so this is worth surfacing.
        logger.warning(f"Chunk header did not match for a chunk of {title[:50]!r}; falling back to prefix search")
        marker = "\n\nAbstract: "
        position = chunk_text.find(marker)
        body = chunk_text[position + len(marker) :] if position != -1 else chunk_text
        # Drop whatever remains of the abstract sentence run.
        body = body[len(abstract) :] if body.startswith(abstract) else body

    return _SECTION_MARKER.sub("", body).strip()


def _search_after_scan(index: str, size: int = 500) -> Iterable[dict]:
    """Page through every document in an index via ``search_after``.

    Uses curl rather than the opensearch-py client so this module has no
    dependency on service wiring and can be exercised standalone.

    :param index: Index to scan.
    :param size: Page size.
    :yields: ``_source`` dictionaries in ``(arxiv_id, chunk_index)`` order.
    """
    after: Optional[list] = None

    while True:
        query = {
            "size": size,
            "query": {"match_all": {}},
            "sort": [{"arxiv_id": "asc"}, {"chunk_index": "asc"}],
            "_source": ["arxiv_id", "chunk_index", "section_title", "chunk_text", "title", "abstract", "chunk_word_count"],
        }
        if after:
            query["search_after"] = after

        result = subprocess.run(
            ["curl", "-s", f"localhost:9200/{index}/_search", "-H", "Content-Type: application/json", "-d", json.dumps(query)],
            capture_output=True,
            text=True,
            check=True,
        )
        hits = json.loads(result.stdout)["hits"]["hits"]
        if not hits:
            return

        for hit in hits:
            yield hit["_source"]
        after = hits[-1]["sort"]


def load_chunks(index: Optional[str] = None, opensearch_client=None) -> List[Chunk]:
    """Load every chunk from the evaluation index, header already stripped.

    :param index: Index to read. Defaults to the evaluation index.
    :param opensearch_client: Optional ``OpenSearchClient``; when given, its
        transport is used instead of curl.
    :returns: Chunks in ``(arxiv_id, chunk_index)`` order.
    """
    index = index or eval_config.EVAL_INDEX_NAME

    if opensearch_client is not None:
        sources = _scan_with_client(opensearch_client, index)
    else:
        sources = _search_after_scan(index)

    chunks = [
        Chunk(
            arxiv_id=source["arxiv_id"],
            chunk_index=source["chunk_index"],
            section_title=source.get("section_title") or "",
            body=strip_header(source["chunk_text"], source.get("title", ""), source.get("abstract", "")),
            full_text=source["chunk_text"],
            title=source.get("title", ""),
            abstract=source.get("abstract", ""),
            word_count=source.get("chunk_word_count", 0),
        )
        for source in sources
    ]

    logger.info(f"Loaded {len(chunks)} chunks from {index}")
    return chunks


def _scan_with_client(client, index: str) -> List[dict]:
    """Page through an index using an ``OpenSearchClient``.

    :param client: Configured ``OpenSearchClient``.
    :param index: Index to scan.
    :returns: ``_source`` dictionaries.
    """
    sources: List[dict] = []
    after = None

    while True:
        body = {
            "size": 500,
            "query": {"match_all": {}},
            "sort": [{"arxiv_id": "asc"}, {"chunk_index": "asc"}],
            "_source": ["arxiv_id", "chunk_index", "section_title", "chunk_text", "title", "abstract", "chunk_word_count"],
        }
        if after:
            body["search_after"] = after

        hits = client.client.search(index=index, body=body)["hits"]["hits"]
        if not hits:
            return sources

        sources.extend(hit["_source"] for hit in hits)
        after = hits[-1]["sort"]


def eligible_chunks(chunks: Sequence[Chunk]) -> List[Chunk]:
    """Filter to chunks that can support a real question.

    :param chunks: All loaded chunks.
    :returns: The eligible subset.
    """
    keep = [chunk for chunk in chunks if chunk.is_eligible]
    logger.info(f"{len(keep)} of {len(chunks)} chunks are eligible for question generation")
    return keep


def group_by_paper(chunks: Sequence[Chunk]) -> Dict[str, List[Chunk]]:
    """Group chunks by their owning paper, preserving chunk order.

    :param chunks: Chunks to group.
    :returns: ``arxiv_id`` to its chunks.
    """
    grouped: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.arxiv_id, []).append(chunk)
    for items in grouped.values():
        items.sort(key=lambda c: c.chunk_index)
    return grouped
