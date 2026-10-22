"""Configuration for the evaluation harness.

Anything that must stay identical between runs for results to remain
comparable lives here rather than being passed around as literals.
"""

from datetime import datetime, timezone
from pathlib import Path

from src.evaluation._bootstrap import REPO_ROOT

# --------------------------------------------------------------------------
# OpenSearch
# --------------------------------------------------------------------------

# Isolated index for the evaluation corpus. Deliberately NOT the production
# `arxiv-papers-chunks` index, so building or wiping the eval corpus can never
# disturb the live one. It uses the same mapping (ARXIV_PAPERS_CHUNKS_MAPPING),
# so retrieval behaves identically to production.
EVAL_INDEX_NAME = "arxiv_evaluation_index"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

# Generates the golden question set (Phase 2) and judges answers (Phase 4).
#
# Qwen3 235B rather than Claude because Anthropic models are not enabled on this
# Bedrock account. Verified to speak the Converse API, so `BedrockClient` is
# reused unchanged, and to emit fence-free JSON deterministically at
# temperature 0 — both are load-bearing for a reproducible dataset.
GENERATOR_MODEL = "qwen.qwen3-vl-235b-a22b"

# The RAG pipeline's own generator. The judge must never be this model: a model
# scoring its own output is not a measurement.
SYSTEM_UNDER_TEST_MODEL = "meta.llama3-70b-instruct-v1:0"

# Pinned so a regenerated dataset is byte-identical to the previous one.
GENERATION_PARAMS = {"temperature": 0.0, "top_p": 1.0}
# 900 truncated a single-hop reply mid-object on a full run, losing that
# question outright — the reference answer plus must_mention plus a verbatim
# answer_snippet adds up. Headroom is cheap; a lost question is not.
GENERATION_MAX_TOKENS = 1500

# Bumped whenever a prompt file changes, so rows record which wording produced
# them and stale rows are identifiable after a prompt edit.
PROMPT_VERSION = "v1"

# --------------------------------------------------------------------------
# Golden dataset targets (120 questions)
# --------------------------------------------------------------------------

# Rebalanced against what this corpus can actually produce, measured over three
# full runs. Comparative questions need two papers in the same topic that share
# a comparable dimension; with 43 papers across 20 topics there are only 30
# same-topic pairings in total, and roughly a quarter of those yield a usable
# question. Chasing 15 meant generating cross-topic pairs that were always
# going to be declined, so the target is set to what the corpus supports and
# the shortfall moved to single-hop, which is not supply-constrained.
TARGET_SINGLE_HOP = 60
TARGET_MULTI_HOP = 20
TARGET_COMPARATIVE = 10
TARGET_UNANSWERABLE = 20
TARGET_PARAPHRASE = 10
TARGET_QUESTIONS = TARGET_SINGLE_HOP + TARGET_MULTI_HOP + TARGET_COMPARATIVE + TARGET_UNANSWERABLE + TARGET_PARAPHRASE

# Chunk eligibility for question generation. Measured against the corpus: these
# leave 696 of 1036 chunks usable, with every paper retaining at least 5.
MIN_CHUNK_BODY_CHARS = 800

# Reject a question when more than this fraction of its content-word trigrams
# also appear in the source chunk. Without it, questions inherit the chunk's
# rare terms and BM25 wins by lexical echo rather than retrieval quality.
MAX_TRIGRAM_LEAKAGE = 0.15

# --------------------------------------------------------------------------
# Retrieval analysis (Phase 3)
# --------------------------------------------------------------------------

# Part of the query-embedding cache key. `JinaEmbeddingsClient.embed_query`
# hardcodes this model/task/dimension; if it changes, the key changes and stale
# vectors are never reused.
QUERY_EMBEDDING_KEY = "jina-embeddings-v3/retrieval.query/1024"

# Uncounted warm-up searches per config before latency is recorded, so the
# first query's connection setup does not land in p95.
RETRIEVAL_WARMUP_QUERIES = 2

# --------------------------------------------------------------------------
# Corpus targets
# --------------------------------------------------------------------------

# Papers must be submitted on or after this date. The evaluation targets
# recent Gen-AI work, and pre-2024 papers predate most of it.
CORPUS_FROM_DATE = "20240101"

# cs.AI: Gen-AI focused. cs.LG: deliberately broad, for topical diversity.
TARGET_CS_AI = 30
TARGET_CS_LG = 20
TARGET_TOTAL = TARGET_CS_AI + TARGET_CS_LG

# Papers whose extracted text is shorter than this are treated as parse
# failures even when docling reported success — a few hundred characters means
# it recovered the title page and little else.
MIN_RAW_TEXT_CHARS = 2000

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

EVAL_ROOT = REPO_ROOT / "src" / "evaluation"
CORPUS_DIR = EVAL_ROOT / "corpus"
CORPUS_MANIFEST = CORPUS_DIR / "corpus_manifest.json"
LOGS_DIR = EVAL_ROOT / "logs"
RESULTS_DIR = EVAL_ROOT / "results"

DATASET_DIR = EVAL_ROOT / "dataset"
DATASETS_DIR = EVAL_ROOT / "datasets"
GOLDEN_DRAFT = DATASETS_DIR / "golden_v1.draft.jsonl"
GOLDEN_SET = DATASETS_DIR / "golden_v1.jsonl"

# Appended to by every retrieval run, never overwritten. The one file under
# results/ that is committed — it is the history that makes runs comparable.
LEADERBOARD = RESULTS_DIR / "leaderboard.md"
# Query vectors keyed by sha256 of question text, so Sweep A embeds each
# question once rather than once per vector/hybrid config.
QUERY_EMBEDDING_CACHE = RESULTS_DIR / "query_embeddings.json"

# PDF download cache for the corpus build.
#
# Two reasons this is set explicitly rather than inherited from
# ``ARXIV__PDF_CACHE_DIR``:
#
# 1. That setting is the *relative* path ``./data/arxiv_pdfs``, resolved against
#    the current working directory. Run the build from anywhere but the repo
#    root and PDFs land in a directory nobody expects — and one that is not
#    covered by the anchored ``data/arxiv_pdfs/`` rule in .gitignore, so a
#    500 MB corpus would be staged for commit.
# 2. The repo-root ``data/arxiv_pdfs`` is owned by UID 50000 (the Airflow
#    container user) and is not group-writable by the host user, so writing
#    there from a host-run build fails with EACCES.
#
# A separate, absolute, host-owned directory avoids both.
PDF_CACHE_DIR = REPO_ROOT / "data" / "eval_pdfs"


def new_run_id() -> str:
    """Timestamped identifier for one harness run.

    :returns: A run id like ``20260809-142301``.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def ensure_dirs() -> None:
    """Create the output directories if they do not exist."""
    for path in (CORPUS_DIR, LOGS_DIR, RESULTS_DIR, PDF_CACHE_DIR, DATASETS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def log_path(run_id: str, name: str) -> Path:
    """Build a path inside the logs directory.

    :param run_id: Identifier from :func:`new_run_id`.
    :param name: Base name, e.g. ``"corpus_build"``.
    :returns: Path like ``logs/corpus_build_20260809-142301.log``.
    """
    return LOGS_DIR / f"{name}_{run_id}.log"
