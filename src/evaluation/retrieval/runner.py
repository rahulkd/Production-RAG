"""Run the retrieval sweep against the evaluation index.

Usage::

    python -m src.evaluation.retrieval.runner                      # full Sweep A (12 configs)
    python -m src.evaluation.retrieval.runner --modes hybrid --top-k 3 10
    python -m src.evaluation.retrieval.runner --limit 5            # smoke test, not added to the leaderboard

From the host rather than a container, point at the published port:
``OPENSEARCH__HOST=http://localhost:9200``.

Calls the search services in-process rather than through ``POST /api/v1/ask``,
so the run needs only OpenSearch and Jina, and ``top_k=20`` is reachable despite
the API's ``le=10`` cap (C4).

Outputs, under ``results/retrieval_<run_id>/``:

* ``retrieval_raw.jsonl`` — per config × question: the ranked list and its
  metrics, so any metric can be re-derived without re-querying.
* ``retrieval_summary.json`` — per config: overall, by question type,
  paraphrase gap, latency.
* ``report.md`` — the same, as tables.

Plus one row per config appended to ``results/leaderboard.md``.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import src.evaluation._bootstrap  # noqa: F401  (must precede src.services imports)
from src.evaluation import config as eval_config
from src.evaluation import report
from src.evaluation.dataset.schema import GoldenQuestion, read_jsonl, validate_dataset
from src.evaluation.logging_setup import setup_logging
from src.evaluation.retrieval import metrics
from src.evaluation.retrieval.configs import MODES, TOP_KS, RetrievalConfig, sweep_a
from src.services.embeddings.factory import make_embeddings_client
from src.services.opensearch.factory import make_opensearch_client_fresh

logger = logging.getLogger(__name__)


def load_questions(path: Path, limit: Optional[int] = None) -> List[GoldenQuestion]:
    """Load the answerable, human-accepted rows of the golden set.

    Unanswerable rows have no ground truth, so retrieval metrics are undefined
    for them; they are scored in Phase 4 on whether the generator refuses.

    :param path: Golden set JSONL.
    :param limit: Keep only the first N answerable rows, for smoke tests.
    :returns: Questions to evaluate.
    :raises ValueError: If the dataset fails schema validation.
    """
    questions = read_jsonl(path, accepted_only=True)

    problems = validate_dataset(questions)
    if problems:
        for problem in problems[:20]:
            logger.error(f"schema problem: {problem}")
        raise ValueError(f"{len(problems)} schema problems in {path}; fix the dataset before scoring against it")

    answerable = [question for question in questions if question.answerable]
    logger.info(f"{len(questions)} accepted questions, {len(answerable)} answerable")
    return answerable[:limit] if limit else answerable


def file_sha256(path: Path) -> str:
    """Hash a file, so the summary records exactly which dataset was scored.

    :param path: File to hash.
    :returns: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QueryEmbeddingCache:
    """Query vectors persisted across runs, keyed by question text.

    Retrieval is deterministic, so the only thing worth caching is the one
    non-free input: the Jina call. The key includes the embedding model and task,
    so a model change can never serve a stale vector.

    :param path: JSON cache file.
    """

    def __init__(self, path: Path):
        self.path = path
        self.vectors: Dict[str, List[float]] = {}
        if path.exists():
            self.vectors = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(question: str) -> str:
        """Cache key for a question.

        :param question: Question text.
        :returns: sha256 of the model key plus the text.
        """
        return hashlib.sha256(f"{eval_config.QUERY_EMBEDDING_KEY}\n{question}".encode("utf-8")).hexdigest()

    def get(self, question: str) -> Optional[List[float]]:
        """Cached vector for a question, if any.

        :param question: Question text.
        :returns: The vector, or None.
        """
        return self.vectors.get(self.key(question))

    def put(self, question: str, vector: List[float]) -> None:
        """Store a vector.

        :param question: Question text.
        :param vector: Its embedding.
        """
        self.vectors[self.key(question)] = vector

    def save(self) -> None:
        """Write the cache to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.vectors), encoding="utf-8")


async def embed_questions(questions: Sequence[GoldenQuestion], cache: QueryEmbeddingCache) -> Dict[str, List[float]]:
    """Embed every question once, reusing cached vectors.

    A failed embedding aborts the run rather than silently dropping the
    question: a missing vector would make vector/hybrid score that question as
    a miss and BM25 would not, which biases exactly the comparison under test.

    :param questions: Questions to embed.
    :param cache: Persistent vector cache.
    :returns: qid to vector.
    """
    vectors: Dict[str, List[float]] = {}
    missing = [question for question in questions if cache.get(question.question) is None]
    logger.info(f"Query embeddings: {len(questions) - len(missing)} cached, {len(missing)} to fetch")

    if missing:
        client = make_embeddings_client()
        try:
            for position, question in enumerate(missing, start=1):
                cache.put(question.question, await client.embed_query(question.question))
                if position % 20 == 0:
                    logger.info(f"  embedded {position}/{len(missing)}")
                    cache.save()
        finally:
            cache.save()
            await client.close()

    for question in questions:
        vectors[question.qid] = cache.get(question.question)
    return vectors


def search(client, retrieval_config: RetrievalConfig, question: str, embedding: Optional[List[float]]) -> List[Dict[str, Any]]:
    """Run one query in one mode.

    These are the calls ``search_unified`` dispatches to, made directly because
    ``search_unified`` and ``search_chunks_vector`` catch every exception and
    return an empty result. Here an OpenSearch error must fail loudly rather
    than be scored as a retrieval miss.

    :param client: ``OpenSearchClient`` bound to the evaluation index.
    :param retrieval_config: Mode and cutoff.
    :param question: Query text.
    :param embedding: Query vector, for vector and hybrid.
    :returns: Hits, best first.
    """
    k = retrieval_config.top_k

    if retrieval_config.mode == "bm25":
        # Same path as search_unified(use_hybrid=False) in /ask.
        return client.search_papers(query=question, size=k, latest=False)["hits"]

    if retrieval_config.mode == "hybrid":
        return client.search_chunks_hybrid(query=question, query_embedding=embedding, size=k, min_score=0.0)["hits"]

    # search_chunks_vector has no non-swallowing variant, so mirror its query.
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": embedding, "k": k}}},
        "_source": {"excludes": ["embedding"]},
    }
    response = client.client.search(index=client.index_name, body=body)
    return [{**hit["_source"], "score": hit["_score"]} for hit in response["hits"]["hits"]]


def ranked_keys(hits: Sequence[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """Convert hits to ground-truth addresses.

    :param hits: Search hits.
    :returns: ``(arxiv_id, chunk_index)`` per hit, in rank order.
    """
    return [(hit["arxiv_id"], int(hit["chunk_index"])) for hit in hits]


def run_config(
    client,
    retrieval_config: RetrievalConfig,
    questions: Sequence[GoldenQuestion],
    vectors: Dict[str, List[float]],
    run_id: str,
) -> List[Dict[str, Any]]:
    """Retrieve and score every question for one config.

    :param client: ``OpenSearchClient`` bound to the evaluation index.
    :param retrieval_config: Mode and cutoff.
    :param questions: Answerable questions.
    :param vectors: qid to query vector.
    :param run_id: Identifier stamped on every row.
    :returns: One raw row per question.
    """
    for question in questions[: eval_config.RETRIEVAL_WARMUP_QUERIES]:
        search(client, retrieval_config, question.question, vectors.get(question.qid))

    rows: List[Dict[str, Any]] = []
    for question in questions:
        started = time.perf_counter()
        hits = search(client, retrieval_config, question.question, vectors.get(question.qid))
        latency_ms = (time.perf_counter() - started) * 1000.0

        ranked = ranked_keys(hits)
        rows.append(
            {
                "run_id": run_id,
                "config": retrieval_config.name,
                "mode": retrieval_config.mode,
                "top_k": retrieval_config.top_k,
                "qid": question.qid,
                "question_type": question.question_type,
                "paraphrase_of": question.paraphrase_of,
                "retrieved": [
                    {"arxiv_id": arxiv_id, "chunk_index": index, "score": hit.get("score")}
                    for (arxiv_id, index), hit in zip(ranked, hits)
                ],
                "metrics": metrics.score_question(ranked, question.relevance_map(), question.relevant_papers, retrieval_config.top_k),
                "latency_ms": round(latency_ms, 2),
            }
        )

    return rows


def paraphrase_gap(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Score drop from an original question to its paraphrase (Q4).

    Same ground truth, different wording. A retriever that matches meaning
    scores both alike; one that matches surface terms drops on the paraphrase.
    Positive gap = the paraphrase scored lower.

    :param rows: Raw rows for one config.
    :returns: Per-metric gap as :func:`metrics.mean_ci`, plus the pair count.
    """
    by_qid = {row["qid"]: row for row in rows}
    pairs = [(by_qid[row["paraphrase_of"]], row) for row in rows if row["paraphrase_of"] in by_qid]

    return {
        "pairs": len(pairs),
        **{
            name: metrics.mean_ci([parent["metrics"][name] - child["metrics"][name] for parent, child in pairs])
            for name in ("recall", "ndcg")
        },
    }


def summarise_config(retrieval_config: RetrievalConfig, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one config's raw rows.

    :param retrieval_config: Mode and cutoff.
    :param rows: Raw rows for that config.
    :returns: Overall, by-type, paraphrase-gap and latency summary.
    """
    by_type: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        by_type.setdefault(row["question_type"], []).append(row["metrics"])

    latencies = np.array([row["latency_ms"] for row in rows], dtype=float)

    return {
        "config": retrieval_config.name,
        "mode": retrieval_config.mode,
        "top_k": retrieval_config.top_k,
        "overall": metrics.aggregate([row["metrics"] for row in rows]),
        "by_type": {question_type: metrics.aggregate(scores) for question_type, scores in sorted(by_type.items())},
        "paraphrase_gap": paraphrase_gap(rows),
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "mean": float(latencies.mean()),
        },
    }


def check_index(client) -> int:
    """Confirm OpenSearch is up and the evaluation index is populated.

    :param client: ``OpenSearchClient`` bound to the evaluation index.
    :returns: Document count.
    :raises RuntimeError: If the cluster is down or the index is empty.
    """
    if not client.health_check():
        raise RuntimeError(f"OpenSearch at {client.host} is not healthy")
    stats = client.get_index_stats()
    if not stats.get("exists") or not stats.get("document_count"):
        raise RuntimeError(f"Index {client.index_name} is missing or empty. Build the corpus first.")
    return stats["document_count"]


async def run(dataset: Path, configs: Sequence[RetrievalConfig], limit: Optional[int], run_id: str) -> int:
    """Run the sweep end to end.

    :param dataset: Golden set JSONL.
    :param configs: Configs to evaluate.
    :param limit: Cap on questions, for smoke tests.
    :param run_id: Run identifier.
    :returns: Process exit code.
    """
    questions = load_questions(dataset, limit)
    if not questions:
        logger.error(f"No answerable accepted questions in {dataset}")
        return 2

    client = make_opensearch_client_fresh(index_name=eval_config.EVAL_INDEX_NAME)
    doc_count = check_index(client)
    logger.info(f"Index {client.index_name}: {doc_count} chunks")

    vectors: Dict[str, List[float]] = {}
    if any(retrieval_config.needs_embedding for retrieval_config in configs):
        vectors = await embed_questions(questions, QueryEmbeddingCache(eval_config.QUERY_EMBEDDING_CACHE))

    out_dir = eval_config.RESULTS_DIR / f"retrieval_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "retrieval_raw.jsonl"

    summaries: List[Dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for retrieval_config in configs:
            rows = run_config(client, retrieval_config, questions, vectors, run_id)
            for row in rows:
                raw_file.write(json.dumps(row) + "\n")

            summary = summarise_config(retrieval_config, rows)
            summaries.append(summary)
            overall = summary["overall"]
            logger.info(
                f"  {retrieval_config.name:<12} R={overall['recall']['mean']:.3f} nDCG={overall['ndcg']['mean']:.3f} "
                f"MRR={overall['mrr']['mean']:.3f} Hit={overall['hit']['mean']:.3f} "
                f"paperR={overall['paper_recall']['mean']:.3f}  p50={summary['latency_ms']['p50']:.0f}ms"
            )

    result = {
        "run_id": run_id,
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "index": client.index_name,
        "index_doc_count": doc_count,
        "questions": len(questions),
        "question_types": {t: sum(1 for q in questions if q.question_type == t) for t in sorted({q.question_type for q in questions})},
        "configs": summaries,
    }
    (out_dir / "retrieval_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report.render_retrieval_report(result), encoding="utf-8")
    logger.info(f"Wrote {out_dir}/ (raw, summary, report)")

    # A partial question set is not comparable with full runs, so it stays out
    # of the history.
    if limit:
        logger.info(f"--limit run: not appended to {eval_config.LEADERBOARD}")
    else:
        report.append_leaderboard(eval_config.LEADERBOARD, result)
        logger.info(f"Appended {len(summaries)} rows to {eval_config.LEADERBOARD}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.retrieval.runner",
        description="Sweep A: score bm25 / vector / hybrid retrieval at several cutoffs against the golden set.",
    )
    parser.add_argument("--dataset", type=Path, default=eval_config.GOLDEN_SET, help="Golden set JSONL")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="Retrieval modes")
    parser.add_argument("--top-k", nargs="+", type=int, default=list(TOP_KS), help="Cutoffs")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N answerable questions (smoke test)")
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
    setup_logging(eval_config.log_path(run_id, "retrieval"), verbose=args.verbose)

    try:
        configs = sweep_a(args.modes, args.top_k)
        return asyncio.run(run(dataset=args.dataset, configs=configs, limit=args.limit, run_id=run_id))
    except KeyboardInterrupt:
        logger.warning("Interrupted; no leaderboard rows written")
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(str(exc))
        return 2
    except Exception as exc:
        logger.exception(f"Retrieval run failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
