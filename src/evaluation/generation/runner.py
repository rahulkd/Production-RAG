"""Run the generation analysis (Phase 4) against the evaluation index.

Usage::

    python -m src.evaluation.generation.runner                     # full run at k=10, appended to the leaderboard
    python -m src.evaluation.generation.runner --top-k 3 5 10      # one leaderboard row per cutoff
    python -m src.evaluation.generation.runner --limit 5           # smoke test, not appended
    python -m src.evaluation.generation.runner --no-judge-cache    # re-judge, ignoring cached verdicts

From the host rather than a container, point at the published port:
``OPENSEARCH__HOST=http://localhost:9200``.

The system under test is what ``/ask`` runs, pinned: hybrid retrieval at each
requested ``top_k`` -> ``RAGPromptBuilder.create_bedrock_structured_prompt``
(C6) -> Llama 3.3 70B at temperature 0 via ``BedrockClient.generate`` (C5) ->
``ResponseParser.parse_structured_response``. It runs in-process so each
answer keeps the ``(arxiv_id, chunk_index)`` of the chunks it was given.

Per answer:

* (a) faithfulness — claims extracted, each judged against the retrieved
  context only; supported / total, with contradictions counted separately.
* (b) relevance — 0-3 against the question and reference answer.
* (c) citations — deterministic; see :mod:`citations`.
* (d) refusal — refused or attempted, scored against ``answerable``.

Outputs, under ``results/generation_<run_id>/<mode>_k<k>/``: ``generation_raw.jsonl``,
``generation_summary.json``, ``report.md``; plus one row per cutoff appended to
``results/generation_leaderboard.md``. The judge is ``JUDGE_MODEL`` (DeepSeek V3.2, temperature 0).
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import src.evaluation._bootstrap  # noqa: F401  (must precede src.services imports)
from src.evaluation import config as eval_config
from src.evaluation import report
from src.evaluation.dataset.schema import GoldenQuestion, read_jsonl, validate_dataset
from src.evaluation.generation.citations import extract_field_citations, score_cited_ids, score_citations
from src.evaluation.generation.judge import Judge, JudgeError, JsonlCache, faithfulness_score, format_context
from src.evaluation.logging_setup import setup_logging
from src.evaluation.retrieval import metrics
from src.evaluation.retrieval.configs import MODES, RetrievalConfig
from src.evaluation.retrieval.runner import QueryEmbeddingCache, check_index, embed_questions, file_sha256, ranked_keys, search
from src.exceptions import BedrockAuthenticationError, BedrockException
from src.services.LLM.factory import make_bedrock_client
from src.services.LLM.prompts import RAGPromptBuilder, ResponseParser
from src.services.opensearch.factory import make_opensearch_client_fresh

logger = logging.getLogger(__name__)

# A run where more questions than this lost a score to an error is not
# comparable with a clean one, so it stays out of the leaderboard.
MAX_FAILED_FRACTION = 0.05

# What /ask returns without calling the LLM when retrieval comes back empty.
NO_CHUNKS_ANSWER = "I couldn't find any relevant information in the papers to answer your question."


def load_questions(path: Path, limit: Optional[int] = None) -> List[GoldenQuestion]:
    """Load every human-accepted row, answerable or not.

    Unlike retrieval, generation scores unanswerable rows too: they are the
    denominator of refusal recall.

    :param path: Golden set JSONL.
    :param limit: Keep only the first N rows, for smoke tests.
    :returns: Questions to evaluate.
    :raises ValueError: If the dataset fails schema validation.
    """
    questions = read_jsonl(path, accepted_only=True)
    problems = validate_dataset(questions)
    if problems:
        for problem in problems[:20]:
            logger.error(f"schema problem: {problem}")
        raise ValueError(f"{len(problems)} schema problems in {path}; fix the dataset before scoring against it")

    logger.info(f"{len(questions)} accepted questions, {sum(q.answerable for q in questions)} answerable")
    return questions[:limit] if limit else questions


def must_mention_rate(answer: str, must_mention: Sequence[str]) -> Optional[float]:
    """Fraction of the golden ``must_mention`` strings present in the answer.

    A cheap deterministic cross-check on the relevance judge.

    :param answer: Generated answer.
    :param must_mention: Literal strings a correct answer should contain.
    :returns: Case-insensitive hit fraction, or None when there are none.
    """
    if not must_mention:
        return None
    lowered = answer.lower()
    return sum(1 for term in must_mention if term.lower() in lowered) / len(must_mention)


class Pipeline:
    """Generates and judges one question at a time.

    :param llm: ``BedrockClient``.
    :param judge: Configured :class:`Judge`.
    :param answer_cache: SUT answers keyed by model, params and prompt.
    :param read_answer_cache: False to regenerate every answer.
    :param concurrency: Questions processed at once.
    """

    def __init__(self, llm, judge: Judge, answer_cache: JsonlCache, read_answer_cache: bool, concurrency: int):
        self.llm = llm
        self.judge = judge
        self.answer_cache = answer_cache
        self.read_answer_cache = read_answer_cache
        self.semaphore = asyncio.Semaphore(concurrency)
        self.prompt_builder = RAGPromptBuilder()
        self.generator_usage = {"inputTokens": 0, "outputTokens": 0}

    async def generate(self, prompt: str) -> Dict[str, Any]:
        """Answer a prompt with the SUT, through the answer cache.

        :param prompt: Structured RAG prompt.
        :returns: ``{"response", "stop_reason"}``.
        """
        max_tokens = self.llm.max_tokens
        key = JsonlCache.key(
            eval_config.SYSTEM_UNDER_TEST_MODEL, json.dumps(eval_config.SUT_PARAMS, sort_keys=True), str(max_tokens), prompt
        )
        cached = self.answer_cache.get(key) if self.read_answer_cache else None
        if cached is not None:
            return cached

        result = await self.llm.generate(
            model=eval_config.SYSTEM_UNDER_TEST_MODEL, prompt=prompt, max_tokens=max_tokens, **eval_config.SUT_PARAMS
        )
        for field_name in self.generator_usage:
            self.generator_usage[field_name] += int(result.get("usage", {}).get(field_name, 0))

        value = {"response": result["response"], "stop_reason": result.get("stop_reason")}
        self.answer_cache.put(key, value)
        return value

    async def run(self, question: GoldenQuestion, hits: Sequence[Dict[str, Any]], top_k: int, run_id: str) -> Dict[str, Any]:
        """Generate, parse, and judge one question.

        Bedrock and judge failures are recorded on the row rather than raised,
        so one bad question does not lose a full run; the summary reports how
        many rows each failure removed.

        :param question: Golden row.
        :param hits: Retrieved chunks, best first.
        :param top_k: Retrieval cutoff.
        :param run_id: Identifier stamped on the row.
        :returns: Raw row.
        """
        async with self.semaphore:
            chunks = [{"arxiv_id": hit["arxiv_id"], "chunk_text": hit["chunk_text"]} for hit in hits]
            ranked = ranked_keys(hits)
            row: Dict[str, Any] = {
                "run_id": run_id,
                "qid": question.qid,
                "question_type": question.question_type,
                "answerable": question.answerable,
                "question": question.question,
                "retrieved": [{"arxiv_id": arxiv_id, "chunk_index": index} for arxiv_id, index in ranked],
                # Exactly what the faithfulness judge saw; calibration shows it to the human labeller.
                "context": format_context(chunks),
                "retrieval": (
                    metrics.score_question(ranked, question.relevance_map(), question.relevant_papers, top_k) if question.answerable else None
                ),
                "judge_errors": [],
            }

            if chunks:
                prompt = self.prompt_builder.create_bedrock_structured_prompt(question.question, chunks)
                row["prompt_chars"] = len(prompt)
                try:
                    generated = await self.generate(prompt)
                except BedrockAuthenticationError:
                    raise
                except BedrockException as exc:
                    row["generation_error"] = str(exc)
                    logger.error(f"{question.qid}: generation failed: {exc}")
                    return row
                raw = generated["response"]
                parsed = ResponseParser.parse_structured_response(raw)
                # The parser falls back to the raw text with empty lists when the
                # model ignored the JSON instruction.
                row["json_ok"] = not (parsed["answer"] == raw and not parsed["sources"] and not parsed["citations"])
                row["stop_reason"] = generated["stop_reason"]
                row["answer_raw"] = raw
                answer = parsed["answer"]
                row["confidence"] = parsed.get("confidence")
                field_citations = extract_field_citations(parsed.get("citations"))
            else:
                row["json_ok"], row["stop_reason"], row["answer_raw"] = None, None, None
                answer = NO_CHUNKS_ANSWER
                field_citations = extract_field_citations(None)

            await self.score_answer(row, question, answer, [chunk["arxiv_id"] for chunk in chunks], field_citations)
            return row

    async def score_answer(
        self,
        row: Dict[str, Any],
        question: GoldenQuestion,
        answer: str,
        retrieved_papers: Sequence[str],
        field_citations: Optional[Dict[str, List[str]]] = None,
        check_faithfulness: bool = True,
    ) -> None:
        """Score one answer in place: citations, must_mention, refusal, faithfulness, relevance.

        Shared with the end-to-end runner, which gets ``answer`` over HTTP
        instead of generating it. ``row["context"]`` must already hold the
        retrieved chunks as the judge should see them.

        :param row: Row under construction; needs ``qid``, ``context`` and ``judge_errors``.
        :param question: Golden row.
        :param answer: The answer text to score.
        :param retrieved_papers: Papers whose chunks were in the prompt.
        :param field_citations: Output of ``extract_field_citations`` for a JSON
            answer; None when the answer has no structured citations field.
        :param check_faithfulness: False when ``row["context"]`` is not known to be
            what the generator saw; faithfulness is then left unscored.
        """
        field_citations = field_citations or extract_field_citations(None)
        row["answer"] = answer
        # Inline [arXiv:id] is the headline: it is what a reader of the answer sees.
        row["citations"] = score_citations(answer, retrieved_papers, question.relevant_papers)
        row["field_citations"] = {
            **score_cited_ids(field_citations["ids"], retrieved_papers, question.relevant_papers),
            "other": field_citations["other"],
        }
        row["must_mention"] = must_mention_rate(answer, question.must_mention) if question.answerable else None

        row["refusal"] = await self._judge(row, "refusal", self.judge.refusal(question.question, answer))
        refused = row["refusal"] is not None and row["refusal"]["label"] == "refused"

        # A refusal asserts nothing to check, and by the rubric's own anchor
        # it scores 0 on relevance, so neither costs a judge call.
        row["claims"], row["faithfulness"] = None, None
        if row["refusal"] is not None and not refused and check_faithfulness:
            claims = await self._judge(row, "claims", self.judge.claims(question.question, answer))
            if claims:
                verdicts = await self._judge(row, "faithfulness", self.judge.verify(row["context"], claims))
                if verdicts is not None:
                    row["claims"], row["faithfulness"] = verdicts, faithfulness_score(verdicts)
            elif claims == []:
                row["claims"], row["faithfulness"] = [], faithfulness_score([])

        row["relevance"] = None
        if question.answerable and row["refusal"] is not None:
            if refused:
                row["relevance"] = {"score": 0, "reason": "refused (scored without a judge call)"}
            else:
                row["relevance"] = await self._judge(
                    row, "relevance", self.judge.relevance(question.question, question.reference_answer, answer)
                )

        logger.info(
            f"  {question.qid} {question.question_type:<12} "
            f"refusal={row['refusal']['label'] if row['refusal'] else 'ERR':<9} "
            f"faith={_fmt_opt(row['faithfulness'] and row['faithfulness']['score'])} "
            f"rel={row['relevance']['score'] if row['relevance'] else '-'} "
            f"cites={len(row['citations']['cited'])} halluc={len(row['citations']['hallucinated'])}"
        )

    async def _judge(self, row: Dict[str, Any], rubric: str, call) -> Any:
        """Await a judge call, recording a failure on the row instead of raising.

        :param row: Row under construction.
        :param rubric: Rubric name, for the error record.
        :param call: The judge coroutine.
        :returns: Its result, or None on failure.
        """
        try:
            return await call
        except BedrockAuthenticationError:
            # Account-wide (no model access, billing): every remaining call would
            # fail the same way, so stop instead of logging it per question.
            raise
        except (JudgeError, BedrockException) as exc:
            row["judge_errors"].append({"rubric": rubric, "error": str(exc)})
            logger.warning(f"{row['qid']}: {rubric} judge failed: {exc}")
            return None


def _fmt_opt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _rate(rows: Sequence[Dict[str, Any]], predicate) -> Dict[str, float]:
    return metrics.mean_ci([1.0 if predicate(row) else 0.0 for row in rows])


def _refused(row: Dict[str, Any]) -> bool:
    return row["refusal"] is not None and row["refusal"]["label"] == "refused"


def _faithful(row: Dict[str, Any]) -> bool:
    """Whether the row has a faithfulness score (an attempted answer with claims)."""
    return bool(row.get("faithfulness")) and row["faithfulness"]["score"] is not None


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate raw rows into the (a)-(d) table plus the Q10 cross-tab.

    No single scalar: faithfulness and refusal trade off against each other,
    and one combined number would hide it.

    :param rows: Raw rows for one run.
    :returns: Summary.
    """
    generated = [row for row in rows if "generation_error" not in row]
    answerable = [row for row in generated if row["answerable"]]
    unanswerable = [row for row in generated if not row["answerable"]]
    classified_answerable = [row for row in answerable if row["refusal"] is not None]
    classified_unanswerable = [row for row in unanswerable if row["refusal"] is not None]
    faithful = [row for row in generated if _faithful(row)]
    scored_relevance = [row for row in answerable if row["relevance"] is not None]

    def faithfulness_block(subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "score": metrics.mean_ci([row["faithfulness"]["score"] for row in subset]),
            "contradicted_per_answer": metrics.mean_ci([row["faithfulness"]["contradicted"] for row in subset]),
            "answers_with_contradiction": _rate(subset, lambda row: row["faithfulness"]["contradicted"] > 0),
            "claims": sum(row["faithfulness"]["claims"] for row in subset),
        }

    by_type: Dict[str, Dict[str, Any]] = {}
    for question_type in sorted({row["question_type"] for row in generated}):
        subset = [row for row in generated if row["question_type"] == question_type]
        by_type[question_type] = {
            "n": len(subset),
            "faithfulness": metrics.mean_ci([row["faithfulness"]["score"] for row in subset if _faithful(row)]),
            "relevance": metrics.mean_ci([row["relevance"]["score"] for row in subset if row["relevance"] is not None]),
            "refused": _rate([row for row in subset if row["refusal"] is not None], _refused),
        }

    # Q10: retrieval upper-bounds generation. Where the answer chunk was never
    # retrieved, a good answer was not possible; where it was and the answer is
    # still poor, generation is the problem.
    bottleneck = {}
    for hit_label, hit_value in (("retrieved", 1.0), ("missed", 0.0)):
        subset = [row for row in scored_relevance if row["retrieval"]["hit"] == hit_value]
        bottleneck[hit_label] = {
            "n": len(subset),
            "answered": sum(1 for row in subset if row["relevance"]["score"] >= 2),
            "relevance": metrics.mean_ci([row["relevance"]["score"] for row in subset]),
            "faithfulness": metrics.mean_ci([row["faithfulness"]["score"] for row in subset if _faithful(row)]),
        }

    def citation_block(key: str) -> Dict[str, Any]:
        scores = [row[key] for row in generated]
        return {
            "precision": metrics.mean_ci([s["precision"] for s in scores if s["precision"] is not None]),
            "recall": metrics.mean_ci([s["recall"] for s in scores if s["recall"] is not None]),
            "present_rate": metrics.mean_ci([s["present"] for s in scores]),
            "hallucinated_total": sum(len(s["hallucinated"]) for s in scores),
            "answers_with_hallucinated": sum(1 for s in scores if s["hallucinated"]),
        }

    return {
        "rows": len(rows),
        "generation_errors": len(rows) - len(generated),
        "json_ok": _rate([row for row in generated if row["json_ok"] is not None], lambda row: row["json_ok"]),
        "truncated": sum(1 for row in generated if row["stop_reason"] in ("max_tokens", "length")),
        "faithfulness": faithfulness_block(faithful),
        "faithfulness_answerable": faithfulness_block([row for row in faithful if row["answerable"]]),
        "faithfulness_unanswerable": faithfulness_block([row for row in faithful if not row["answerable"]]),
        "relevance": metrics.mean_ci([row["relevance"]["score"] for row in scored_relevance]),
        "answered_rate": _rate(scored_relevance, lambda row: row["relevance"]["score"] >= 2),
        "must_mention": metrics.mean_ci([row["must_mention"] for row in answerable if row["must_mention"] is not None]),
        "citations": citation_block("citations"),
        "field_citations": {
            **citation_block("field_citations"),
            "answers_with_non_arxiv_entries": sum(1 for row in generated if row["field_citations"]["other"]),
        },
        "refusal": {
            "recall": _rate(classified_unanswerable, _refused),
            "false_refusal_rate": _rate(classified_answerable, _refused),
        },
        "by_type": by_type,
        "bottleneck": bottleneck,
        "judge_errors": sum(len(row["judge_errors"]) for row in rows),
    }


def retrieve_all(client, retrieval_config: RetrievalConfig, questions: Sequence[GoldenQuestion], vectors: Dict[str, List[float]]):
    """Retrieve chunks for every question up front.

    OpenSearch calls are fast and synchronous; doing them first keeps the
    concurrent phase purely Bedrock-bound.

    :param client: ``OpenSearchClient`` bound to the evaluation index.
    :param retrieval_config: Mode and cutoff.
    :param questions: Questions.
    :param vectors: qid to query vector.
    :returns: qid to hits.
    """
    return {question.qid: search(client, retrieval_config, question.question, vectors.get(question.qid)) for question in questions}


def _usage_snapshot(pipeline: "Pipeline", judge: Judge) -> Dict[str, Any]:
    return {
        "generator": dict(pipeline.generator_usage),
        "judge": dict(judge.usage),
        "judge_calls": judge.calls,
        "judge_cache_hits": judge.cache_hits,
    }


def _usage_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: {field: after[key][field] - before[key][field] for field in after[key]} if isinstance(after[key], dict) else after[key] - before[key]
        for key in after
    }


async def evaluate_cutoff(
    args: argparse.Namespace,
    run_id: str,
    retrieval_config: RetrievalConfig,
    questions: Sequence[GoldenQuestion],
    hits_by_qid: Dict[str, List[Dict[str, Any]]],
    pipeline: "Pipeline",
    judge: Judge,
    index_info: Dict[str, Any],
) -> bool:
    """Generate, judge and report one retrieval cutoff.

    :param args: Parsed CLI arguments.
    :param run_id: Run identifier.
    :param retrieval_config: Mode and cutoff.
    :param questions: Questions.
    :param hits_by_qid: Retrieved chunks per question at this cutoff.
    :param pipeline: Shared generator/judge pipeline.
    :param judge: Shared judge (its counters are cumulative across cutoffs).
    :param index_info: ``{"index", "index_doc_count"}``.
    :returns: False if too many questions failed to be scored.
    """
    logger.info(f"=== {retrieval_config.name}: {len(questions)} questions ===")
    before = _usage_snapshot(pipeline, judge)
    rows = await asyncio.gather(*(pipeline.run(question, hits_by_qid[question.qid], retrieval_config.top_k, run_id) for question in questions))

    out_dir = eval_config.RESULTS_DIR / f"generation_{run_id}" / f"{retrieval_config.mode}_k{retrieval_config.top_k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "generation_raw.jsonl").open("w", encoding="utf-8") as raw_file:
        for row in rows:
            raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "run_id": run_id,
        "dataset": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        **index_info,
        "retrieval": retrieval_config.name,
        "sut_model": eval_config.SYSTEM_UNDER_TEST_MODEL,
        "sut_params": eval_config.SUT_PARAMS,
        "judge_model": judge.model,
        "judge_params": judge.params,
        "judge_prompt_version": judge.prompt_version,
        "judge_cache_read": judge.read_cache,
        "questions": len(questions),
        "question_types": {t: sum(1 for q in questions if q.question_type == t) for t in sorted({q.question_type for q in questions})},
        "usage": _usage_delta(before, _usage_snapshot(pipeline, judge)),
        "summary": summarise(rows),
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report.render_generation_report(result), encoding="utf-8")
    logger.info(f"Wrote {out_dir}/ (raw, summary, report)")

    summary = result["summary"]
    logger.info(
        f"{retrieval_config.name}: faithfulness={summary['faithfulness']['score']['mean']:.3f} relevance={summary['relevance']['mean']:.2f} "
        f"cite_precision={summary['citations']['precision']['mean']:.3f} refusal_recall={summary['refusal']['recall']['mean']:.3f} "
        f"false_refusal={summary['refusal']['false_refusal_rate']['mean']:.3f} "
        f"judge_calls={result['usage']['judge_calls']} cache_hits={result['usage']['judge_cache_hits']} errors={summary['judge_errors']}"
    )

    failed = summary["generation_errors"] + sum(1 for row in rows if row["judge_errors"])
    if args.limit:
        logger.info(f"--limit run: not appended to {eval_config.GENERATION_LEADERBOARD}")
    elif failed > MAX_FAILED_FRACTION * len(rows):
        logger.error(
            f"{retrieval_config.name}: {failed}/{len(rows)} questions hit a generation or judge error; not appended to "
            f"{eval_config.GENERATION_LEADERBOARD}. Re-run: successful judgments are cached."
        )
        return False
    else:
        report.append_generation_leaderboard(eval_config.GENERATION_LEADERBOARD, result)
        logger.info(f"Appended {retrieval_config.name} to {eval_config.GENERATION_LEADERBOARD}")
    return True


async def run(args: argparse.Namespace, run_id: str) -> int:
    """Run the generation analysis end to end, once per requested cutoff.

    :param args: Parsed CLI arguments.
    :param run_id: Run identifier.
    :returns: Process exit code.
    """
    if args.judge_model == eval_config.SYSTEM_UNDER_TEST_MODEL:
        raise ValueError(f"Judge model {args.judge_model} is the system under test; a model cannot grade its own answers")
    if args.judge_model == eval_config.GENERATOR_MODEL:
        logger.warning(f"Judge {args.judge_model} also wrote the reference answers; relevance scores will favour its phrasing")

    questions = load_questions(args.dataset, args.limit)
    if not questions:
        logger.error(f"No accepted questions in {args.dataset}")
        return 2

    configs = [RetrievalConfig(mode=args.mode, top_k=k) for k in sorted(set(args.top_k))]
    client = make_opensearch_client_fresh(index_name=eval_config.EVAL_INDEX_NAME)
    doc_count = check_index(client)
    logger.info(f"Index {client.index_name}: {doc_count} chunks; retrieval {', '.join(c.name for c in configs)}")

    vectors: Dict[str, List[float]] = {}
    if configs[0].needs_embedding:
        vectors = await embed_questions(questions, QueryEmbeddingCache(eval_config.QUERY_EMBEDDING_CACHE))

    llm = make_bedrock_client()
    judge = Judge(
        llm,
        model=args.judge_model,
        params=eval_config.JUDGE_PARAMS,
        max_tokens=eval_config.JUDGE_MAX_TOKENS,
        prompt_version=eval_config.JUDGE_PROMPT_VERSION,
        cache=JsonlCache(eval_config.JUDGE_CACHE),
        read_cache=not args.no_judge_cache,
    )
    pipeline = Pipeline(
        llm, judge, JsonlCache(eval_config.ANSWER_CACHE), read_answer_cache=not args.regenerate, concurrency=args.concurrency
    )
    logger.info(f"SUT {eval_config.SYSTEM_UNDER_TEST_MODEL} {eval_config.SUT_PARAMS}; judge {judge.model} {judge.params}")

    index_info = {"index": client.index_name, "index_doc_count": doc_count}
    all_ok = True
    for retrieval_config in configs:
        # Each k is a real query at size=k, not a truncation of the largest:
        # hybrid's RRF fusion can rank the top 3 differently at size=3 and size=10.
        hits_by_qid = retrieve_all(client, retrieval_config, questions, vectors)
        all_ok &= await evaluate_cutoff(args, run_id, retrieval_config, questions, hits_by_qid, pipeline, judge, index_info)
    return 0 if all_ok else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.generation.runner",
        description="Phase 4: score faithfulness, relevance, citations and refusals of the RAG generator.",
    )
    parser.add_argument("--dataset", type=Path, default=eval_config.GOLDEN_REVIEWED, help="Golden set JSONL (accepted/edited rows only)")
    parser.add_argument("--mode", choices=MODES, default=eval_config.GENERATION_RETRIEVAL_MODE, help="Retrieval mode")
    parser.add_argument(
        "--top-k", nargs="+", type=int, default=list(eval_config.GENERATION_TOP_KS), help="Chunks passed to the generator; several values compare cutoffs"
    )
    parser.add_argument("--judge-model", default=eval_config.JUDGE_MODEL, help="Bedrock model id for the judge")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke test)")
    parser.add_argument("--concurrency", type=int, default=eval_config.GENERATION_CONCURRENCY, help="Questions in flight")
    parser.add_argument("--no-judge-cache", action="store_true", help="Ignore cached verdicts (determinism check)")
    parser.add_argument("--regenerate", action="store_true", help="Ignore cached SUT answers")
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
    setup_logging(eval_config.log_path(run_id, "generation"), verbose=args.verbose)

    try:
        return asyncio.run(run(args, run_id))
    except KeyboardInterrupt:
        logger.warning("Interrupted; no leaderboard row written (paid judgments are kept in the cache)")
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, BedrockAuthenticationError) as exc:
        logger.error(str(exc))
        return 2
    except Exception as exc:
        logger.exception(f"Generation run failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
