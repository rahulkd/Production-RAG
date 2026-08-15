"""The corpus manifest: the frozen, version-controlled definition of the corpus.

The manifest is what makes evaluation numbers reproducible. It pins the exact
arXiv IDs rather than a query, because re-running a query returns different
papers every day, and it records a hash of each paper's extracted text.

The hash matters more than it looks. Ground truth in Phase 2 is addressed as
``(arxiv_id, chunk_index)`` — a *positional* label whose meaning depends on the
extracted text and the chunking being byte-identical to when it was recorded.
If arXiv serves a new version, or docling changes its extraction, chunk 7
becomes different text, every label silently points somewhere wrong, and the
scores stay plausible. Asserting the hash on rebuild converts that silent
corruption into a loud failure.

A manifest moves through two states:

* **discovered** — IDs and metadata chosen, ``raw_text_sha256`` still null.
* **built** — text hashes and chunk counts filled in by the build step.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


def text_sha256(text: str) -> str:
    """Hash extracted paper text.

    :param text: The paper's extracted plain text.
    :returns: Lowercase hex SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ManifestPaper:
    """One paper's entry in the manifest.

    :param arxiv_id: Full ID including version suffix, as arXiv returned it.
    :param base_id: Version-stripped ID; the stable dedup key.
    :param title: Paper title.
    :param primary_category: First (primary) arXiv category.
    :param categories: All arXiv categories.
    :param published_date: Submission date as arXiv reported it.
    :param topic_key: Topic slice that selected this paper.
    :param via_backfill: True if it filled another topic's shortfall.
    :param raw_text_sha256: Hash of the extracted text; null until built.
    :param raw_text_chars: Length of the extracted text; null until built.
    :param n_chunks: Chunks produced and indexed; null until built.
    """

    arxiv_id: str
    base_id: str
    title: str
    primary_category: str
    categories: List[str]
    published_date: str
    topic_key: str
    via_backfill: bool = False
    raw_text_sha256: Optional[str] = None
    raw_text_chars: Optional[int] = None
    n_chunks: Optional[int] = None

    @property
    def is_built(self) -> bool:
        """Whether the build step has filled in this paper's text hash."""
        return self.raw_text_sha256 is not None


@dataclass
class CorpusManifest:
    """The full manifest.

    :param manifest_version: Schema version of this file.
    :param created_at: UTC timestamp of the discovery run.
    :param built_at: UTC timestamp of the most recent successful build.
    :param date_window: ``from``/``to`` bounds used at discovery, ``YYYYMMDD``.
    :param index_name: OpenSearch index the corpus is built into.
    :param embedding_model: Embedding model, recorded because changing it
        invalidates comparisons.
    :param embedding_dim: Embedding dimensionality.
    :param chunking: Chunker settings in force at build time.
    :param papers: The pinned papers.
    """

    manifest_version: int = MANIFEST_VERSION
    created_at: str = ""
    built_at: Optional[str] = None
    date_window: Dict[str, str] = field(default_factory=dict)
    index_name: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    chunking: Dict[str, int] = field(default_factory=dict)
    papers: List[ManifestPaper] = field(default_factory=list)

    # ---------------------------------------------------------------- access

    @property
    def arxiv_ids(self) -> List[str]:
        """Full arXiv IDs, in manifest order."""
        return [paper.arxiv_id for paper in self.papers]

    @property
    def is_built(self) -> bool:
        """True when every paper carries a text hash."""
        return bool(self.papers) and all(paper.is_built for paper in self.papers)

    def by_base_id(self) -> Dict[str, ManifestPaper]:
        """Index the papers by their version-stripped ID.

        :returns: Base ID to manifest entry.
        """
        return {paper.base_id: paper for paper in self.papers}

    def category_counts(self) -> Dict[str, int]:
        """Papers per primary category.

        :returns: Primary category to count.
        """
        counts: Dict[str, int] = {}
        for paper in self.papers:
            counts[paper.primary_category] = counts.get(paper.primary_category, 0) + 1
        return counts

    def topic_counts(self) -> Dict[str, int]:
        """Papers per topic slice.

        :returns: Topic key to count.
        """
        counts: Dict[str, int] = {}
        for paper in self.papers:
            counts[paper.topic_key] = counts.get(paper.topic_key, 0) + 1
        return counts

    # ------------------------------------------------------------------- io

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-ready dictionary.

        :returns: The manifest as nested plain data.
        """
        payload = asdict(self)
        payload["papers"] = [asdict(paper) for paper in self.papers]
        return payload

    def save(self, path: Path) -> None:
        """Write the manifest as pretty-printed JSON.

        :param path: Destination file. Parent dirs are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        logger.info(f"Wrote manifest with {len(self.papers)} papers to {path}")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CorpusManifest":
        """Build a manifest from parsed JSON.

        :param payload: Parsed manifest JSON.
        :returns: The manifest.
        :raises ValueError: If the schema version is unsupported.
        """
        version = payload.get("manifest_version")
        if version != MANIFEST_VERSION:
            raise ValueError(f"Unsupported manifest_version {version!r}, expected {MANIFEST_VERSION}")

        papers = [ManifestPaper(**entry) for entry in payload.get("papers", [])]
        fields = {key: value for key, value in payload.items() if key != "papers"}
        return cls(papers=papers, **fields)

    @classmethod
    def load(cls, path: Path) -> "CorpusManifest":
        """Read a manifest from disk.

        :param path: Manifest file.
        :returns: The manifest.
        :raises FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"No manifest at {path}. Run `python -m src.evaluation.corpus.discover` first.")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class HashCheck:
    """Result of verifying a rebuilt paper against its recorded hash.

    :param arxiv_id: The paper checked.
    :param expected: Hash recorded in the manifest.
    :param actual: Hash of the text just extracted.
    """

    arxiv_id: str
    expected: str
    actual: str

    @property
    def matches(self) -> bool:
        """Whether the extracted text is byte-identical to the recorded run."""
        return self.expected == self.actual


def verify_text_hashes(manifest: CorpusManifest, extracted: Dict[str, str]) -> List[HashCheck]:
    """Compare freshly extracted text against the manifest's recorded hashes.

    Papers with no recorded hash (first build) are skipped rather than
    reported as mismatches.

    :param manifest: Manifest holding the expected hashes.
    :param extracted: Base arXiv ID to freshly extracted text.
    :returns: Only the checks that failed; empty means the corpus is intact.
    """
    failures: List[HashCheck] = []

    for paper in manifest.papers:
        if paper.raw_text_sha256 is None:
            continue
        text = extracted.get(paper.base_id)
        if text is None:
            continue

        actual = text_sha256(text)
        if actual != paper.raw_text_sha256:
            failures.append(HashCheck(arxiv_id=paper.arxiv_id, expected=paper.raw_text_sha256, actual=actual))

    return failures
