"""Pre-flight check: how many corpus papers will the PDF parser reject?

Downloads every PDF in the manifest and measures page count and file size —
but does **not** parse them. Docling parsing is the slow part (~25s for an
8-page paper, ~75s for 12 pages), so this answers "how many papers will I
lose, and to which limit?" in minutes instead of hours.

Why it matters: ``PDFParserSettings`` caps pages and file size, and a paper
over either cap is stored metadata-only with empty ``raw_text``. The build
then skips it — and **a skipped paper is not replaced**, so the corpus ends
below target with the affected topic under quota. Surveys and papers with long
appendices are the usual casualties.

Downloads land in the shared evaluation PDF cache, so a subsequent
``corpus.build`` reuses them instead of re-fetching.

Usage::

    python -m src.evaluation.corpus.preflight
    python -m src.evaluation.corpus.preflight --limit 10
    python -m src.evaluation.corpus.preflight --max-pages 60   # test a proposed limit
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pypdfium2
import src.evaluation._bootstrap  # noqa: F401  (must precede src.services imports)
from src.config import get_settings
from src.evaluation import config as eval_config
from src.evaluation.corpus.manifest import CorpusManifest, ManifestPaper
from src.evaluation.logging_setup import setup_logging
from src.schemas.arxiv.paper import ArxivPaper

logger = logging.getLogger(__name__)

# arXiv PDF URLs are deterministic from the versioned ID, so the manifest does
# not need to carry them.
PDF_URL_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}"

# Matches the production ingestion path (metadata_fetcher uses 5).
DEFAULT_CONCURRENCY = 5


@dataclass
class PdfStats:
    """Measurements for one paper's PDF.

    :param arxiv_id: The paper's arXiv ID.
    :param title: Paper title, for readable reports.
    :param topic_key: Topic slice that selected it, so a shortfall can be
        traced to the slice it will damage.
    :param pages: Page count, or None if it could not be measured.
    :param size_mb: File size in megabytes, or None if not downloaded.
    :param error: Why measurement failed, when it did.
    """

    arxiv_id: str
    title: str
    topic_key: str
    pages: Optional[int] = None
    size_mb: Optional[float] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the PDF was downloaded and measured."""
        return self.pages is not None and self.error == ""

    def verdict(self, max_pages: int, max_size_mb: int) -> str:
        """Classify against the parser's limits.

        :param max_pages: Page cap.
        :param max_size_mb: File size cap in MB.
        :returns: ``OK``, ``OVER_PAGES``, ``OVER_SIZE``, or ``ERROR``.
        """
        if not self.ok:
            return "ERROR"
        if self.pages > max_pages:
            return "OVER_PAGES"
        if self.size_mb is not None and self.size_mb > max_size_mb:
            return "OVER_SIZE"
        return "OK"


def count_pages(path: Path) -> int:
    """Count pages in a PDF.

    Uses pypdfium2, which is already present as docling's PDF backend, so the
    count matches what the parser itself would see.

    :param path: Local PDF file.
    :returns: Number of pages.
    :raises Exception: If the file cannot be opened as a PDF.
    """
    document = pypdfium2.PdfDocument(path)
    try:
        return len(document)
    finally:
        document.close()


def as_arxiv_paper(entry: ManifestPaper) -> ArxivPaper:
    """Build the minimal ``ArxivPaper`` that ``download_pdf`` requires.

    Only ``arxiv_id`` and ``pdf_url`` are actually read by the downloader; the
    rest are required by the schema and carry no meaning here.

    :param entry: Manifest entry.
    :returns: A stub paper suitable for downloading.
    """
    return ArxivPaper(
        arxiv_id=entry.arxiv_id,
        title=entry.title,
        authors=[],
        abstract="",
        categories=list(entry.categories),
        published_date=entry.published_date,
        pdf_url=PDF_URL_TEMPLATE.format(arxiv_id=entry.arxiv_id),
    )


async def measure(entry: ManifestPaper, arxiv_client, semaphore: asyncio.Semaphore) -> PdfStats:
    """Download one PDF and measure it.

    :param entry: Manifest entry to measure.
    :param arxiv_client: Client with the evaluation PDF cache configured.
    :param semaphore: Limits concurrent downloads.
    :returns: The measurement, with ``error`` set if it could not be taken.
    """
    stats = PdfStats(arxiv_id=entry.arxiv_id, title=entry.title, topic_key=entry.topic_key)

    async with semaphore:
        try:
            path = await arxiv_client.download_pdf(as_arxiv_paper(entry), False)
        except Exception as exc:
            stats.error = f"download failed: {exc}"
            logger.error(f"  {entry.arxiv_id}: {stats.error}")
            return stats

    if path is None or not path.exists():
        stats.error = "download returned no file"
        logger.error(f"  {entry.arxiv_id}: {stats.error}")
        return stats

    stats.size_mb = path.stat().st_size / 1_000_000

    try:
        # Blocking C call; keep the event loop free for the other downloads.
        stats.pages = await asyncio.to_thread(count_pages, path)
    except Exception as exc:
        stats.error = f"unreadable PDF: {exc}"
        logger.error(f"  {entry.arxiv_id}: {stats.error}")

    return stats


def histogram(results: Sequence[PdfStats], max_pages: int) -> List[str]:
    """Render a page-count distribution.

    :param results: Successful measurements.
    :param max_pages: Current cap, marked in the output.
    :returns: Report lines.
    """
    buckets = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 60), (61, 100), (101, 10_000)]
    lines = ["", "Page-count distribution", "-" * 46]

    for low, high in buckets:
        count = sum(1 for r in results if r.ok and low <= r.pages <= high)
        if count == 0:
            continue
        label = f"{low}-{high}" if high < 10_000 else f"{low}+"
        marker = "  <-- over the limit" if low > max_pages else ""
        lines.append(f"  {label:>8} pages  {'#' * count:<20} {count:>3}{marker}")

    return lines


def summarise(results: Sequence[PdfStats], max_pages: int, max_size_mb: int) -> List[str]:
    """Build the human-readable report.

    :param results: All measurements.
    :param max_pages: Page cap in force.
    :param max_size_mb: File size cap in force.
    :returns: Report lines.
    """
    measured = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]

    over_pages = [r for r in measured if r.pages > max_pages]
    over_size = [r for r in measured if r.size_mb is not None and r.size_mb > max_size_mb]
    rejected = {r.arxiv_id for r in over_pages} | {r.arxiv_id for r in over_size}

    lines = [
        "",
        "=" * 70,
        f"PRE-FLIGHT: {len(results)} papers, {len(measured)} measured, {len(errors)} unreadable",
        f"Limits in force: max_pages={max_pages}, max_file_size_mb={max_size_mb}",
        "=" * 70,
    ]

    if measured:
        pages = sorted(r.pages for r in measured)
        sizes = sorted(r.size_mb for r in measured if r.size_mb is not None)
        lines += [
            "",
            f"  pages   min={pages[0]}  median={pages[len(pages) // 2]}  max={pages[-1]}",
            f"  size MB min={sizes[0]:.1f}  median={sizes[len(sizes) // 2]:.1f}  max={sizes[-1]:.1f}",
        ]
        lines += histogram(measured, max_pages)

    lines += ["", f"WOULD BE SKIPPED: {len(rejected)} of {len(results)}", "-" * 46]

    if over_pages:
        lines.append(f"  over {max_pages} pages ({len(over_pages)}):")
        for r in sorted(over_pages, key=lambda x: -x.pages):
            lines.append(f"    {r.arxiv_id:<15} {r.pages:>4}p  [{r.topic_key}] {r.title[:44]}")
    if over_size:
        lines.append(f"  over {max_size_mb} MB ({len(over_size)}):")
        for r in sorted(over_size, key=lambda x: -(x.size_mb or 0)):
            lines.append(f"    {r.arxiv_id:<15} {r.size_mb:>5.1f}MB  [{r.topic_key}] {r.title[:42]}")
    if errors:
        lines.append(f"  unreadable ({len(errors)}):")
        for r in errors:
            lines.append(f"    {r.arxiv_id:<15} {r.error}")
    if not rejected and not errors:
        lines.append("  none — every paper fits the current limits")

    # Which topics lose papers, and by how much. A shortfall concentrated in
    # one slice hurts the corpus more than the same count spread thinly.
    if rejected:
        by_topic: dict = {}
        for r in results:
            if r.arxiv_id in rejected:
                by_topic[r.topic_key] = by_topic.get(r.topic_key, 0) + 1
        lines += ["", "Topic slices affected:", "-" * 46]
        for key, count in sorted(by_topic.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {key:<28} -{count}")

    # What raising the cap would buy.
    if measured:
        lines += ["", "Coverage if max_pages were raised:", "-" * 46]
        current_ok = sum(1 for r in measured if r.pages <= max_pages)
        lines.append(f"    {max_pages:>4} (current)   {current_ok:>3}/{len(measured)} papers")
        for candidate in (40, 50, 60, 80, 100):
            if candidate <= max_pages:
                continue
            covered = sum(1 for r in measured if r.pages <= candidate)
            gain = covered - current_ok
            suffix = f"   (+{gain})" if gain else ""
            lines.append(f"    {candidate:>4}             {covered:>3}/{len(measured)} papers{suffix}")

    return lines


async def preflight(manifest_path: Path, limit: Optional[int], concurrency: int, max_pages: Optional[int]) -> int:
    """Run the pre-flight check.

    :param manifest_path: Manifest to check.
    :param limit: Measure only the first N papers.
    :param concurrency: Simultaneous downloads.
    :param max_pages: Page cap to evaluate against; defaults to the configured
        ``PDF_PARSER__MAX_PAGES``.
    :returns: Process exit code. 0 when nothing would be skipped, 4 otherwise.
    """
    # Imported here so `--help` works without touching the arXiv settings.
    from src.evaluation.corpus.build import make_eval_arxiv_client

    settings = get_settings()
    effective_max_pages = max_pages if max_pages is not None else settings.pdf_parser.max_pages
    max_size_mb = settings.pdf_parser.max_file_size_mb

    manifest = CorpusManifest.load(manifest_path)
    entries = manifest.papers[:limit] if limit else manifest.papers

    logger.info(f"Pre-flight on {len(entries)} papers (download only, no parsing)")
    logger.info(f"PDF cache: {eval_config.PDF_CACHE_DIR}  (reused by corpus.build)")
    logger.info(f"Checking against max_pages={effective_max_pages}, max_file_size_mb={max_size_mb}")

    arxiv_client = make_eval_arxiv_client()
    semaphore = asyncio.Semaphore(concurrency)

    results = await asyncio.gather(*(measure(entry, arxiv_client, semaphore) for entry in entries))

    report = summarise(results, effective_max_pages, max_size_mb)
    for line in report:
        logger.info(line)

    report_path = eval_config.LOGS_DIR / f"preflight_{eval_config.new_run_id()}.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info(f"\nReport written to {report_path}")

    would_skip = sum(1 for r in results if r.verdict(effective_max_pages, max_size_mb) != "OK")
    return 0 if would_skip == 0 else 4


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.corpus.preflight",
        description="Download corpus PDFs (no parsing) and report the page-count distribution.",
    )
    parser.add_argument("--manifest", type=Path, default=eval_config.CORPUS_MANIFEST, help="Manifest to check")
    parser.add_argument("--limit", type=int, default=None, help="Measure only the first N papers")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Simultaneous downloads")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Page cap to evaluate against (default: PDF_PARSER__MAX_PAGES). Use to test a proposed limit.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG output on the console")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Process exit code.
    """
    args = parse_args(argv)
    eval_config.ensure_dirs()

    run_id = eval_config.new_run_id()
    setup_logging(eval_config.log_path(run_id, "preflight"), verbose=args.verbose)

    try:
        return asyncio.run(
            preflight(
                manifest_path=args.manifest,
                limit=args.limit,
                concurrency=args.concurrency,
                max_pages=args.max_pages,
            )
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted; PDFs downloaded so far remain cached")
        return 130
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    except Exception as exc:
        logger.exception(f"Pre-flight failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
