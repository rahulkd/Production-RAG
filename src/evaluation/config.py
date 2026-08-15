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
    for path in (CORPUS_DIR, LOGS_DIR, RESULTS_DIR, PDF_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def log_path(run_id: str, name: str) -> Path:
    """Build a path inside the logs directory.

    :param run_id: Identifier from :func:`new_run_id`.
    :param name: Base name, e.g. ``"corpus_build"``.
    :returns: Path like ``logs/corpus_build_20260809-142301.log``.
    """
    return LOGS_DIR / f"{name}_{run_id}.log"
