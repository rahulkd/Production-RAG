"""Run the end-to-end analysis (Phase 5): the golden set over HTTP.

Usage::

    python -m src.evaluation.e2e.runner                                  # /ask and /stream at k=3 and k=10
    python -m src.evaluation.e2e.runner --endpoints stream --top-k 3     # what the UI sends
    python -m src.evaluation.e2e.runner --limit 5                        # smoke test, not appended

Point ``--base-url`` at an API running with ``OPENSEARCH__ACTIVE_INDEX=evaluation``.
The golden ground truth lives in the frozen evaluation index; an API on the
(empty) production index answers every question with "no relevant information",
which the preflight detects. See the README for the command.

Phases 3-4 call the services in-process. This phase sends the same questions
through the real routes, so it measures what a client gets:

* **Q11 agreement** — the API's ``sources`` against the papers the in-process
  harness retrieves for the same query and k. A mismatch means the route's
  retrieval diverges from what Phases 3-4 measured (e.g. ``ask.py``'s silent
  BM25 fallback when the Jina call fails).
* **search_mode** — must read ``hybrid`` on every response.
* **latency** — total, plus time to first token on ``/stream``.
* **answer quality** — the Phase 4 judge and citation checks, on the answer as
  served. The API does not return chunk text, so the judge's context is the
  harness retrieval for the same query; faithfulness is only scored where the
  API's papers match it.

``/ask`` uses the structured JSON prompt; ``/stream`` — the only route the UI
calls — uses the plain prompt. Both hardcode temperature 0.7 (C5), so answers
are one sample each, not pinned.

Outputs, under ``results/e2e_<run_id>/<endpoint>_k<k>/``: ``e2e_raw.jsonl``,
``e2e_summary.json``; plus ``results/e2e_<run_id>/report.md`` comparing every
config, and one row per config appended to ``results/e2e_leaderboard.md``.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
import numpy as np

import src.evaluation._bootstrap  # noqa: F401  (must precede src.services imports)
from src.evaluation import config as eval_config
from src.evaluation import report
from src.evaluation.dataset.schema import GoldenQuestion
from src.evaluation.generation.citations import normalize_id
from src.evaluation.generation.judge import Judge, JsonlCache, format_context
from src.evaluation.generation.runner import Pipeline, load_questions, summarise
from src.evaluation.logging_setup import setup_logging
from src.evaluation.retrieval import metrics
from src.evaluation.retrieval.configs import RetrievalConfig
from src.evaluation.retrieval.runner import QueryEmbeddingCache, check_index, embed_questions, file_sha256, ranked_keys, search
from src.exceptions import BedrockAuthenticationError
from src.services.LLM.factory import make_bedrock_client
from src.services.opensearch.factory import make_opensearch_client_fresh

logger = logging.getLogger(__name__)

ENDPOINTS = ("ask", "stream")

# Canned replies the routes return without calling the LLM when retrieval is empty.
NO_CHUNK_ANSWERS = (
    "I couldn't find any relevant information in the papers to answer your question.",
    "No relevant information found.",
)

_PDF_URL_ID = re.compile(r"arxiv\.org/pdf/([^/]+?)(?:\.pdf)?$")


def paper_ids_from_sources(sources: Sequence[str]) -> List[str]:
    """Normalized arXiv IDs from the API's PDF-URL sources.

    :param sources: e.g. ``["https://arxiv.org/pdf/2401.06831.pdf"]``.
    :returns: Version-stripped IDs, in order, deduplicated.
    """
    ids: List[str] = []
    for url in sources:
        match = _PDF_URL_ID.search(url)
        arxiv_id = normalize_id(match.group(1) if match else url)
        if arxiv_id not in ids:
            ids.append(arxiv_id)
    return ids


def parse_stream(text: str) -> Dict[str, Any]:
    """Reassemble a ``/stream`` response from its ``data:`` frames.

    Frames, in order: metadata (``sources``, ``chunks_used``, ``search_mode``),
    any number of ``chunk`` frames, then ``{"answer", "done"}``. An ``error``
    frame can replace any of them.

    :param text: Full response body.
    :returns: ``answer``, ``sources``, ``chunks_used``, ``search_mode``, ``error``.
    """
    result: Dict[str, Any] = {"answer": None, "sources": [], "chunks_used": None, "search_mode": None, "error": None}
    pieces: List[str] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        frame = json.loads(line[len("data: ") :])
        if "error" in frame:
            result["error"] = frame["error"]
        if "sources" in frame:
            result["sources"] = frame["sources"]
        for key in ("chunks_used", "search_mode"):
            if key in frame:
                result[key] = frame[key]
        if "chunk" in frame:
            pieces.append(frame["chunk"])
        if frame.get("done"):
            result["answer"] = frame.get("answer", "".join(pieces))
    if result["answer"] is None and pieces:
        result["answer"] = "".join(pieces)
    return result


class ApiClient:
    """Calls one RAG route and times it.

    :param base_url: API root, e.g. ``http://localhost:8001``.
    :param timeout: Per-request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float = 300.0):
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def close(self) -> None:
        """Close the underlying connection pool."""
        await self.client.aclose()

    async def call(self, endpoint: str, question: str, top_k: int) -> Dict[str, Any]:
        """Send one question.

        :param endpoint: ``ask`` or ``stream``.
        :param question: Query text.
        :param top_k: Chunks the route should retrieve.
        :returns: Parsed response plus ``latency_s``, ``ttft_s`` (stream only), ``http_status``.
        """
        payload = {"query": question, "top_k": top_k, "use_hybrid": True, "model": eval_config.SYSTEM_UNDER_TEST_MODEL}
        started = time.perf_counter()

        if endpoint == "ask":
            response = await self.client.post("/api/v1/ask", json=payload)
            latency = time.perf_counter() - started
            if response.status_code != 200:
                return {"http_status": response.status_code, "error": response.text[:300], "latency_s": latency}
            body = response.json()
            return {**body, "http_status": 200, "error": None, "latency_s": latency, "ttft_s": None}

        ttft: Optional[float] = None
        parts: List[str] = []
        async with self.client.stream("POST", "/api/v1/stream", json=payload) as response:
            if response.status_code != 200:
                return {"http_status": response.status_code, "error": (await response.aread()).decode()[:300], "latency_s": time.perf_counter() - started}
            async for text in response.aiter_text():
                if ttft is None and '"chunk"' in text:
                    ttft = time.perf_counter() - started
                parts.append(text)
        parsed = parse_stream("".join(parts))
        return {**parsed, "http_status": 200, "latency_s": time.perf_counter() - started, "ttft_s": ttft}


async def preflight(api: ApiClient, question: GoldenQuestion) -> None:
    """Fail fast when the API cannot answer from the evaluation corpus.

    :param api: Client.
    :param question: An answerable golden question.
    :raises RuntimeError: If the API is down or its index returns nothing.
    """
    try:
        health = await api.client.get("/api/v1/health")
    except httpx.HTTPError as exc:
        raise RuntimeError(f"API at {api.client.base_url} is unreachable: {exc}") from exc
    if health.status_code != 200:
        raise RuntimeError(f"API health check returned {health.status_code}: {health.text[:200]}")

    probe = await api.call("ask", question.question, 3)
    if probe.get("error") or not probe.get("chunks_used"):
        raise RuntimeError(
            f"API retrieved no chunks for a known-answerable question ({question.qid}); its index does not hold the "
            f"evaluation corpus. Response: {json.dumps({k: probe.get(k) for k in ('answer', 'chunks_used', 'error')})[:300]}"
        )


async def evaluate_question(
    api: ApiClient,
    pipeline: Pipeline,
    endpoint: str,
    question: GoldenQuestion,
    hits: Sequence[Dict[str, Any]],
    top_k: int,
    run_id: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Call the API for one question and score the served answer.

    :param api: Client.
    :param pipeline: Phase 4 pipeline, used only for :meth:`Pipeline.score_answer`.
    :param endpoint: ``ask`` or ``stream``.
    :param question: Golden row.
    :param hits: In-process hybrid retrieval for the same query and k.
    :param top_k: Cutoff sent to the API.
    :param run_id: Identifier stamped on the row.
    :param semaphore: Bounds requests in flight.
    :returns: Raw row.
    """
    async with semaphore:
        chunks = [{"arxiv_id": hit["arxiv_id"], "chunk_text": hit["chunk_text"]} for hit in hits]
        ranked = ranked_keys(hits)
        harness_papers = list(dict.fromkeys(normalize_id(chunk["arxiv_id"]) for chunk in chunks))
        row: Dict[str, Any] = {
            "run_id": run_id,
            "endpoint": endpoint,
            "top_k": top_k,
            "qid": question.qid,
            "question_type": question.question_type,
            "answerable": question.answerable,
            "question": question.question,
            "harness_papers": harness_papers,
            "context": format_context(chunks),
            "retrieval": metrics.score_question(ranked, question.relevance_map(), question.relevant_papers, top_k) if question.answerable else None,
            "judge_errors": [],
            "stop_reason": None,
        }

        try:
            served = await api.call(endpoint, question.question, top_k)
        except httpx.HTTPError as exc:
            served = {"http_status": None, "error": f"{type(exc).__name__}: {exc}", "latency_s": None}
        row.update(
            http_status=served.get("http_status"),
            latency_s=served.get("latency_s"),
            ttft_s=served.get("ttft_s"),
            search_mode=served.get("search_mode"),
            chunks_used=served.get("chunks_used"),
            sources=served.get("sources") or [],
        )
        if served.get("error") or served.get("answer") is None:
            row["generation_error"] = served.get("error") or "no answer in response"
            logger.error(f"{endpoint} k={top_k} {question.qid}: {row['generation_error']}")
            return row

        answer = served["answer"]
        api_papers = paper_ids_from_sources(row["sources"])
        row["api_papers"] = api_papers
        row["papers_agree"] = set(api_papers) == set(harness_papers)
        row["no_chunk_answer"] = answer.strip() in NO_CHUNK_ANSWERS
        relevant = {normalize_id(paper) for paper in question.relevant_papers}
        row["api_paper_recall"] = len(relevant & set(api_papers)) / len(relevant) if question.answerable else None
        # ResponseParser's fallback hands back the raw model JSON when parsing fails (F8).
        row["json_ok"] = not answer.lstrip().startswith("{") if endpoint == "ask" else None

        await pipeline.score_answer(row, question, answer, api_papers, check_faithfulness=row["papers_agree"])
        return row


def _percentiles(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    present = np.array([value for value in values if value is not None], dtype=float)
    if not present.size:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    return {"p50": float(np.percentile(present, 50)), "p95": float(np.percentile(present, 95)), "mean": float(present.mean()), "n": int(present.size)}


def summarise_e2e(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Phase 5 metrics plus the Phase 4 quality summary for one config.

    :param rows: Raw rows for one endpoint and k.
    :returns: Summary.
    """
    served = [row for row in rows if "generation_error" not in row]
    answerable = [row for row in served if row["answerable"]]
    modes: Dict[str, int] = {}
    for row in served:
        modes[str(row["search_mode"])] = modes.get(str(row["search_mode"]), 0) + 1

    return {
        "http_errors": len(rows) - len(served),
        "search_modes": modes,
        "papers_agree": metrics.mean_ci([1.0 if row["papers_agree"] else 0.0 for row in served]),
        "no_chunk_answers": sum(1 for row in served if row["no_chunk_answer"]),
        "api_paper_recall": metrics.mean_ci([row["api_paper_recall"] for row in answerable]),
        "harness_paper_recall": metrics.mean_ci([row["retrieval"]["paper_recall"] for row in answerable]),
        "latency_s": _percentiles([row["latency_s"] for row in served]),
        "ttft_s": _percentiles([row["ttft_s"] for row in served]),
        "quality": summarise(rows),
    }


async def run(args: argparse.Namespace, run_id: str) -> int:
    """Run every requested endpoint at every requested cutoff.

    :param args: Parsed CLI arguments.
    :param run_id: Run identifier.
    :returns: Process exit code.
    """
    questions = load_questions(args.dataset, args.limit)
    if not questions:
        logger.error(f"No accepted questions in {args.dataset}")
        return 2

    api = ApiClient(args.base_url)
    try:
        await preflight(api, next(question for question in questions if question.answerable))
        logger.info(f"API {args.base_url} answers from a populated index")

        client = make_opensearch_client_fresh(index_name=eval_config.EVAL_INDEX_NAME)
        doc_count = check_index(client)
        vectors = await embed_questions(questions, QueryEmbeddingCache(eval_config.QUERY_EMBEDDING_CACHE))

        llm = make_bedrock_client()
        judge = Judge(
            llm,
            model=eval_config.JUDGE_MODEL,
            params=eval_config.JUDGE_PARAMS,
            max_tokens=eval_config.JUDGE_MAX_TOKENS,
            prompt_version=eval_config.JUDGE_PROMPT_VERSION,
            cache=JsonlCache(eval_config.JUDGE_CACHE),
        )
        pipeline = Pipeline(llm, judge, JsonlCache(eval_config.ANSWER_CACHE), read_answer_cache=True, concurrency=args.concurrency)
        semaphore = asyncio.Semaphore(args.concurrency)

        run_dir = eval_config.RESULTS_DIR / f"e2e_{run_id}"
        results: List[Dict[str, Any]] = []
        for top_k in sorted(set(args.top_k)):
            # Same retrieval the harness would do for this query and k: the
            # reference the API's sources are compared against.
            retrieval_config = RetrievalConfig(mode="hybrid", top_k=top_k)
            hits_by_qid = {q.qid: search(client, retrieval_config, q.question, vectors.get(q.qid)) for q in questions}

            for endpoint in args.endpoints:
                logger.info(f"=== /{endpoint} k={top_k}: {len(questions)} questions ===")
                rows = await asyncio.gather(
                    *(evaluate_question(api, pipeline, endpoint, q, hits_by_qid[q.qid], top_k, run_id, semaphore) for q in questions)
                )
                out_dir = run_dir / f"{endpoint}_k{top_k}"
                out_dir.mkdir(parents=True, exist_ok=True)
                with (out_dir / "e2e_raw.jsonl").open("w", encoding="utf-8") as raw_file:
                    for row in rows:
                        raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")

                result = {
                    "run_id": run_id,
                    "endpoint": endpoint,
                    "top_k": top_k,
                    "base_url": args.base_url,
                    "dataset": str(args.dataset),
                    "dataset_sha256": file_sha256(args.dataset),
                    "index": client.index_name,
                    "index_doc_count": doc_count,
                    "sut_model": eval_config.SYSTEM_UNDER_TEST_MODEL,
                    "judge_model": judge.model,
                    "judge_prompt_version": judge.prompt_version,
                    "questions": len(questions),
                    "summary": summarise_e2e(rows),
                }
                (out_dir / "e2e_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                results.append(result)

                summary = result["summary"]
                quality = summary["quality"]
                logger.info(
                    f"/{endpoint} k={top_k}: agree={summary['papers_agree']['mean']:.3f} modes={summary['search_modes']} "
                    f"p50={summary['latency_s']['p50']}s faithfulness={quality['faithfulness']['score']['mean']:.3f} "
                    f"relevance={quality['relevance']['mean']:.2f} refusal_recall={quality['refusal']['recall']['mean']:.3f} "
                    f"http_errors={summary['http_errors']} judge_errors={quality['judge_errors']}"
                )
    finally:
        await api.close()

    (run_dir / "report.md").write_text(report.render_e2e_report(results), encoding="utf-8")
    logger.info(f"Wrote {run_dir}/report.md")
    if args.limit:
        logger.info(f"--limit run: not appended to {eval_config.E2E_LEADERBOARD}")
    else:
        report.append_e2e_leaderboard(eval_config.E2E_LEADERBOARD, results)
        logger.info(f"Appended {len(results)} rows to {eval_config.E2E_LEADERBOARD}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.e2e.runner", description="Phase 5: the golden set through the live API routes.")
    parser.add_argument("--base-url", default=eval_config.E2E_BASE_URL, help="API root URL")
    parser.add_argument("--endpoints", nargs="+", choices=ENDPOINTS, default=list(ENDPOINTS), help="Routes to evaluate")
    parser.add_argument("--top-k", nargs="+", type=int, default=list(eval_config.E2E_TOP_KS), help="Cutoffs to send (API caps at 10)")
    parser.add_argument("--dataset", type=Path, default=eval_config.GOLDEN_REVIEWED, help="Golden set JSONL (accepted/edited rows only)")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke test)")
    parser.add_argument("--concurrency", type=int, default=eval_config.GENERATION_CONCURRENCY, help="Requests in flight")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG output on the console")
    args = parser.parse_args(argv)
    if any(k < 1 or k > 10 for k in args.top_k):
        parser.error("--top-k values must be 1-10 (AskRequest caps top_k at 10)")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Process exit code.
    """
    args = parse_args(argv)
    eval_config.ensure_dirs()
    run_id = eval_config.new_run_id()
    setup_logging(eval_config.log_path(run_id, "e2e"), verbose=args.verbose)

    try:
        return asyncio.run(run(args, run_id))
    except KeyboardInterrupt:
        logger.warning("Interrupted; no leaderboard rows written (paid judgments are kept in the cache)")
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, BedrockAuthenticationError) as exc:
        logger.error(str(exc))
        return 2
    except Exception as exc:
        logger.exception(f"E2E run failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
