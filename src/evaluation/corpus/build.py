"""Build the evaluation corpus from the manifest.

Takes the pinned arXiv IDs and drives them through the production ingestion
path — fetch, download, docling parse, store to Postgres — then chunks,
embeds, and indexes them into the isolated evaluation index.

Usage::

    python -m src.evaluation.corpus.build
    python -m src.evaluation.corpus.build --force-index   # recreate the index first
    python -m src.evaluation.corpus.build --skip-fetch    # reindex from Postgres only
    python -m src.evaluation.corpus.build --limit 3       # smoke test on 3 papers

Two properties this step is responsible for:

* **Failures are visible.** The ingestion path treats download and parse
  errors as non-fatal, so a paper can be stored with no text and the run still
  looks green. Every such paper is collected and written to a dedicated
  failure log, and echoed to the terminal.
* **The corpus is frozen.** Each paper's extracted text is hashed into the
  manifest. On a rebuild the hashes are re-checked, so a changed docling
  version or a new arXiv version fails loudly instead of silently
  invalidating the ground-truth chunk labels recorded in Phase 2.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import src.evaluation._bootstrap  # noqa: F401  (must precede src.services imports)
from src.config import get_settings
from src.database import get_db_session
from src.evaluation import config as eval_config
from src.evaluation.corpus.manifest import CorpusManifest, text_sha256, verify_text_hashes
from src.evaluation.logging_setup import FailureRecorder, setup_logging
from src.models.paper import Paper
from src.services.arxiv.client import ArxivClient
from src.services.indexing.factory import make_hybrid_indexing_service
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.opensearch.factory import make_opensearch_client_fresh
from src.services.pdf_parser.factory import make_pdf_parser_service

logger = logging.getLogger(__name__)


def make_eval_arxiv_client() -> "ArxivClient":
    """Build an arXiv client whose PDF cache is an absolute, host-owned path.

    ``make_arxiv_client()`` inherits ``ARXIV__PDF_CACHE_DIR``, a *relative*
    path resolved against the current working directory, and one that points
    at a directory owned by the Airflow container user. See the note on
    :data:`src.evaluation.config.PDF_CACHE_DIR` for why neither is usable here.

    :returns: A client that caches PDFs under ``data/eval_pdfs``.
    """
    settings = get_settings()
    arxiv_settings = settings.arxiv.model_copy(update={"pdf_cache_dir": str(eval_config.PDF_CACHE_DIR)})
    return ArxivClient(settings=arxiv_settings)


def positive_int(value: str) -> int:
    """argparse type for a count that must be at least 1.

    :param value: Raw argument text.
    :returns: The parsed integer.
    :raises argparse.ArgumentTypeError: If it is not a positive integer.
    """
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def base_id_of(arxiv_id: str) -> str:
    """Strip a trailing version marker from an arXiv ID.

    :param arxiv_id: ID with or without a version suffix.
    :returns: The version-stripped ID.
    """
    from src.evaluation.corpus.selection import base_arxiv_id

    return base_arxiv_id(arxiv_id)


def paper_to_index_payload(paper: Paper) -> Dict[str, Any]:
    """Convert a stored ``Paper`` row into the dict the indexer expects.

    ``published_date`` is rendered as an ISO string because the OpenSearch
    mapping declares it a ``date`` field and the bulk helper serialises plain
    strings more predictably than datetimes.

    :param paper: ORM row.
    :returns: Payload for ``HybridIndexingService.index_paper``.
    """
    return {
        "id": str(paper.id),
        "arxiv_id": paper.arxiv_id,
        "title": paper.title or "",
        "abstract": paper.abstract or "",
        "authors": paper.authors or [],
        "categories": paper.categories or [],
        "published_date": paper.published_date.isoformat() if paper.published_date else None,
        "raw_text": paper.raw_text or "",
        "sections": paper.sections,
    }


def load_papers_by_base_id(session, base_ids: Sequence[str]) -> Dict[str, Paper]:
    """Load the manifest's papers from Postgres, keyed by version-stripped ID.

    Matching is on the base ID because arXiv may serve a different version
    than the one recorded at discovery, which would make an exact ``arxiv_id``
    lookup miss.

    :param session: Open SQLAlchemy session.
    :param base_ids: Version-stripped IDs to look for.
    :returns: Base ID to the stored row.
    """
    wanted = set(base_ids)
    found: Dict[str, Paper] = {}

    for paper in session.query(Paper).all():
        key = base_id_of(paper.arxiv_id)
        if key in wanted:
            found[key] = paper

    return found


async def fetch_and_store(manifest: CorpusManifest, failures: FailureRecorder, limit: int | None) -> None:
    """Fetch the manifest's papers and store them in Postgres.

    :param manifest: The corpus manifest.
    :param failures: Recorder for per-paper failures.
    :param limit: Process only the first N papers, for smoke tests.
    """
    arxiv_ids = manifest.arxiv_ids[:limit] if limit else manifest.arxiv_ids
    titles = {paper.base_id: paper.title for paper in manifest.papers}

    logger.info(f"Fetching {len(arxiv_ids)} papers from arXiv (rate-limited ~3s each)")
    logger.info(f"PDF cache: {eval_config.PDF_CACHE_DIR}")

    arxiv_client = make_eval_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    fetcher = make_metadata_fetcher(arxiv_client, pdf_parser)

    with get_db_session() as session:
        results = await fetcher.fetch_and_process_papers(
            arxiv_ids=list(arxiv_ids),
            process_pdfs=True,
            store_to_db=True,
            db_session=session,
        )

    logger.info(
        f"Ingestion: fetched={results['papers_fetched']} downloaded={results['pdfs_downloaded']} "
        f"parsed={results['pdfs_parsed']} stored={results['papers_stored']}"
    )

    for arxiv_id in results.get("not_found", []):
        failures.record(arxiv_id, "fetch", "arXiv returned no entry for this ID", titles.get(base_id_of(arxiv_id), ""))
    for arxiv_id in results.get("download_failures", []):
        failures.record(arxiv_id, "download", "PDF download failed", titles.get(base_id_of(arxiv_id), ""))
    for arxiv_id in results.get("parse_failures", []):
        failures.record(arxiv_id, "parse", "docling returned no content", titles.get(base_id_of(arxiv_id), ""))


async def index_papers(
    manifest: CorpusManifest,
    failures: FailureRecorder,
    force_index: bool,
    limit: int | None,
) -> Dict[str, int]:
    """Chunk, embed, and index the corpus into the evaluation index.

    Papers with missing or implausibly short extracted text are recorded as
    failures and skipped — indexing a title page as if it were a paper would
    quietly poison the retrieval metrics.

    :param manifest: The corpus manifest; updated in place with text hashes
        and chunk counts.
    :param failures: Recorder for per-paper failures.
    :param force_index: Delete and recreate the index before indexing.
    :param limit: Process only the first N papers.
    :returns: Aggregate counters for the run.
    """
    index_name = manifest.index_name or eval_config.EVAL_INDEX_NAME
    settings = get_settings()

    opensearch = make_opensearch_client_fresh(settings, index_name=index_name)
    if not opensearch.health_check():
        raise RuntimeError(f"OpenSearch is not reachable at {settings.opensearch.host}")

    logger.info(f"Preparing index {index_name!r} (force={force_index})")
    setup = opensearch.setup_indices(force=force_index)
    logger.info(f"  index created={setup['hybrid_index']}, rrf pipeline created={setup['rrf_pipeline']}")

    indexer = make_hybrid_indexing_service(settings, index_name=index_name)

    entries = manifest.papers[:limit] if limit else manifest.papers
    base_ids = [entry.base_id for entry in entries]

    stats = {"indexed": 0, "skipped": 0, "chunks": 0, "hash_mismatches": 0}
    extracted: Dict[str, str] = {}

    with get_db_session() as session:
        stored = load_papers_by_base_id(session, base_ids)

        for entry in entries:
            paper = stored.get(entry.base_id)

            if paper is None:
                failures.record(entry.arxiv_id, "missing_in_db", "Not found in Postgres after ingestion", entry.title)
                stats["skipped"] += 1
                continue

            raw_text = paper.raw_text or ""
            if len(raw_text) < eval_config.MIN_RAW_TEXT_CHARS:
                failures.record(
                    entry.arxiv_id,
                    "empty_text",
                    f"Extracted text is {len(raw_text)} chars, below the {eval_config.MIN_RAW_TEXT_CHARS} minimum",
                    entry.title,
                )
                stats["skipped"] += 1
                continue

            extracted[entry.base_id] = raw_text

            # Drop any chunks this paper already has before re-indexing it.
            # Without this a re-run (e.g. after a transient embedding failure)
            # appends a second copy of every chunk, which silently inflates
            # recall and corrupts precision in the retrieval metrics.
            opensearch.delete_paper_chunks(paper.arxiv_id)

            result = await indexer.index_paper(paper_to_index_payload(paper))

            if result["chunks_indexed"] == 0:
                failures.record(
                    entry.arxiv_id,
                    "index",
                    f"No chunks indexed (created={result['chunks_created']}, errors={result['errors']})",
                    entry.title,
                )
                stats["skipped"] += 1
                continue

            entry.raw_text_sha256 = text_sha256(raw_text)
            entry.raw_text_chars = len(raw_text)
            entry.n_chunks = result["chunks_indexed"]

            stats["indexed"] += 1
            stats["chunks"] += result["chunks_indexed"]
            logger.info(f"  indexed {entry.arxiv_id}: {result['chunks_indexed']} chunks")

    # Freeze check: on a rebuild, extracted text must match what was recorded.
    mismatches = verify_text_hashes(manifest, extracted)
    for mismatch in mismatches:
        failures.record(
            mismatch.arxiv_id,
            "hash_mismatch",
            f"Extracted text changed since the last build "
            f"(expected {mismatch.expected[:12]}…, got {mismatch.actual[:12]}…). "
            "Ground-truth chunk labels for this paper are stale.",
        )
    stats["hash_mismatches"] = len(mismatches)

    return stats


async def build(
    manifest_path: Path,
    force_index: bool,
    skip_fetch: bool,
    limit: int | None,
    run_id: str,
) -> int:
    """Run the corpus build end to end.

    :param manifest_path: Manifest to build from; updated in place on success.
    :param force_index: Delete and recreate the evaluation index first.
    :param skip_fetch: Skip arXiv fetching and index whatever is in Postgres.
    :param limit: Process only the first N papers.
    :param run_id: Identifier used to name the log files.
    :returns: Process exit code.
    """
    manifest = CorpusManifest.load(manifest_path)
    logger.info(f"Manifest: {len(manifest.papers)} papers, window {manifest.date_window}")
    logger.info(f"  primary categories: {manifest.category_counts()}")
    if manifest.is_built:
        logger.info("  manifest is already built — text hashes will be re-verified")

    failures = FailureRecorder()

    if skip_fetch:
        logger.info("Skipping arXiv fetch (--skip-fetch); indexing from Postgres only")
    else:
        await fetch_and_store(manifest, failures, limit)

    stats = await index_papers(manifest, failures, force_index, limit)

    if stats["indexed"]:
        manifest.built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest.save(manifest_path)

    failure_log = eval_config.log_path(run_id, "pdf_failures")
    failures.write(failure_log, failure_log.with_suffix(".json"))

    logger.info("=== Build summary ===")
    logger.info(f"  papers indexed : {stats['indexed']}")
    logger.info(f"  papers skipped : {stats['skipped']}")
    logger.info(f"  chunks indexed : {stats['chunks']}")
    logger.info(f"  failures       : {len(failures)}  -> {failure_log}")

    if failures:
        logger.error(f"{len(failures)} paper(s) failed. Breakdown by stage:")
        for stage, items in sorted(failures.by_stage().items()):
            logger.error(f"  {stage}: {len(items)} -> {', '.join(f.arxiv_id for f in items)}")

    if stats["hash_mismatches"]:
        logger.error(
            f"{stats['hash_mismatches']} paper(s) changed since the last build. "
            "The corpus is NOT frozen; ground-truth labels for those papers must be regenerated."
        )
        return 3

    if stats["indexed"] == 0:
        logger.error("No papers were indexed.")
        return 1

    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.corpus.build",
        description="Fetch, parse, and index the pinned evaluation corpus into an isolated OpenSearch index.",
    )
    parser.add_argument("--manifest", type=Path, default=eval_config.CORPUS_MANIFEST, help="Manifest to build from")
    parser.add_argument("--force-index", action="store_true", help="Delete and recreate the evaluation index first")
    parser.add_argument("--skip-fetch", action="store_true", help="Index from Postgres without calling arXiv")
    # positive_int rather than plain int: `--limit 0` is falsy, so it would slip
    # through the `if limit` guards below and silently process the whole corpus.
    parser.add_argument("--limit", type=positive_int, default=None, help="Process only the first N papers (smoke test)")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG output on the console")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Process exit code.
    """
    args = parse_args(argv)
    eval_config.ensure_dirs()

    run_id = eval_config.new_run_id()
    setup_logging(eval_config.log_path(run_id, "corpus_build"), verbose=args.verbose)

    try:
        return asyncio.run(
            build(
                manifest_path=args.manifest,
                force_index=args.force_index,
                skip_fetch=args.skip_fetch,
                limit=args.limit,
                run_id=run_id,
            )
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted — partial progress is in Postgres and the index")
        return 130
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    except Exception as exc:
        logger.exception(f"Build failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
