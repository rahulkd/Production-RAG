"""Freeze and unfreeze the evaluation index.

The golden set labels chunks as ``(arxiv_id, chunk_index)``, so any write to the
evaluation index can silently move a label. Freezing sets OpenSearch's
``index.blocks.write``: indexing, updates and deletes of documents are refused
by the cluster itself, whoever sends them.

Usage::

    python -m src.evaluation.corpus.freeze            # verify against the manifest, then freeze
    python -m src.evaluation.corpus.freeze --status   # report only
    python -m src.evaluation.corpus.freeze --lift     # unfreeze, e.g. before a deliberate rebuild

Freezing refuses an index that does not match the manifest, so a half-built
corpus cannot be frozen by mistake. A write block does not stop the index being
*deleted*; :mod:`src.evaluation.corpus.build` therefore checks the freeze itself
and needs ``--unfreeze`` to touch a frozen index.

From the host rather than a container: ``OPENSEARCH__HOST=http://localhost:9200``.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import src.evaluation._bootstrap  # noqa: F401  (must precede src.services imports)
from src.evaluation import config as eval_config
from src.evaluation.corpus.manifest import CorpusManifest
from src.evaluation.logging_setup import setup_logging
from src.services.opensearch.factory import make_opensearch_client_fresh

logger = logging.getLogger(__name__)

WRITE_BLOCK = "index.blocks.write"


def is_frozen(client: Any, index_name: str) -> bool:
    """Whether the index refuses document writes.

    :param client: Raw ``opensearchpy.OpenSearch`` client.
    :param index_name: Index to check.
    :returns: True when ``index.blocks.write`` is set. False if the index is absent.
    """
    if not client.indices.exists(index=index_name):
        return False
    settings = client.indices.get_settings(index=index_name, flat_settings=True)
    return str(settings[index_name]["settings"].get(WRITE_BLOCK, "false")).lower() == "true"


def verify_against_manifest(client: Any, index_name: str, manifest: CorpusManifest) -> Dict[str, int]:
    """Check the index holds exactly what the manifest says was built.

    :param client: Raw OpenSearch client.
    :param index_name: Index to check.
    :param manifest: Built manifest.
    :returns: ``{"chunks", "papers"}`` counts, equal to the manifest's.
    :raises RuntimeError: If the manifest is unbuilt or the counts differ.
    """
    # Not manifest.is_built: that demands a text hash for *every* paper, and papers
    # whose PDF failed to download or parse legitimately never get one.
    if not manifest.built_at or not any(paper.n_chunks for paper in manifest.papers):
        raise RuntimeError("Manifest records no completed build; run `python -m src.evaluation.corpus.build` first")
    if not client.indices.exists(index=index_name):
        raise RuntimeError(f"Index {index_name!r} does not exist")

    client.indices.refresh(index=index_name)
    chunks = client.count(index=index_name)["count"]
    papers = client.search(index=index_name, body={"size": 0, "aggs": {"p": {"cardinality": {"field": "arxiv_id"}}}})["aggregations"]["p"]["value"]

    expected_chunks = sum(paper.n_chunks or 0 for paper in manifest.papers)
    expected_papers = sum(1 for paper in manifest.papers if paper.n_chunks)
    if (chunks, papers) != (expected_chunks, expected_papers):
        raise RuntimeError(
            f"Index {index_name!r} holds {chunks} chunks from {papers} papers; the manifest expects "
            f"{expected_chunks} chunks from {expected_papers} papers. Not freezing."
        )
    return {"chunks": chunks, "papers": papers}


def freeze(client: Any, index_name: str, manifest: CorpusManifest) -> Dict[str, int]:
    """Verify the index against the manifest, then block writes to it.

    :param client: Raw OpenSearch client.
    :param index_name: Index to freeze.
    :param manifest: Built manifest.
    :returns: The verified counts.
    """
    counts = verify_against_manifest(client, index_name, manifest)
    client.indices.put_settings(index=index_name, body={WRITE_BLOCK: True})
    return counts


def lift(client: Any, index_name: str) -> None:
    """Allow document writes to the index again.

    :param client: Raw OpenSearch client.
    :param index_name: Index to unfreeze.
    """
    client.indices.put_settings(index=index_name, body={WRITE_BLOCK: False})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.corpus.freeze", description="Freeze or unfreeze the evaluation index.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="Report whether the index is frozen")
    mode.add_argument("--lift", action="store_true", help="Unfreeze the index (re-run without flags to freeze again)")
    parser.add_argument("--manifest", type=Path, default=eval_config.CORPUS_MANIFEST, help="Manifest to verify against")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Process exit code.
    """
    args = parse_args(argv)
    eval_config.ensure_dirs()
    run_id = eval_config.new_run_id()
    setup_logging(eval_config.log_path(run_id, "corpus_freeze"))

    index_name = eval_config.EVAL_INDEX_NAME
    client = make_opensearch_client_fresh(index_name=index_name).client
    try:
        if args.status:
            logger.info(f"{index_name}: {'FROZEN' if is_frozen(client, index_name) else 'writable'}")
            return 0
        if args.lift:
            lift(client, index_name)
            logger.warning(f"{index_name} is writable again. Re-run without flags once the rebuild is verified.")
            return 0
        counts = freeze(client, index_name, CorpusManifest.load(args.manifest))
    except (RuntimeError, FileNotFoundError) as exc:
        logger.error(str(exc))
        return 2
    logger.info(f"{index_name} FROZEN: {counts['chunks']} chunks from {counts['papers']} papers match the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
