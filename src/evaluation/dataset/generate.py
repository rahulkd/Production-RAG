"""Generate the golden question set.

Reads the evaluation index, samples seed chunks, drives Qwen over the prompt
templates, filters leaky questions, and writes a draft JSONL for human review.

Usage::

    python -m src.evaluation.dataset.generate
    python -m src.evaluation.dataset.generate --limit 5      # smoke test
    python -m src.evaluation.dataset.generate --types single_hop

Nothing here writes the final dataset. Output goes to ``golden_v1.draft.jsonl``;
:mod:`src.evaluation.dataset.review` promotes accepted rows to
``golden_v1.jsonl``. An unreviewed LLM-generated set measures the generator, not
the retriever.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import src.evaluation._bootstrap  # noqa: F401  (must precede src.services imports)
from src.evaluation import config as eval_config
from src.evaluation.corpus.manifest import CorpusManifest, text_sha256
from src.evaluation.corpus.topics import ALL_TOPICS
from src.evaluation.dataset import leakage as leak
from src.evaluation.dataset.chunks import Chunk, eligible_chunks, group_by_paper, load_chunks
from src.evaluation.dataset.sample import PairSeed, coverage_report, sample_pairs, sample_single_hop
from src.evaluation.dataset.schema import (
    GRADE_ANSWER,
    GRADE_SUPPORTING,
    GoldenQuestion,
    RelevantChunk,
    validate_dataset,
    write_jsonl,
)
from src.evaluation.logging_setup import setup_logging
from src.services.LLM.factory import make_bedrock_client

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Body text sent to the generator per excerpt. Long enough to contain a real
# claim, short enough to keep two excerpts plus instructions inside a sane
# prompt for the pair types.
MAX_BODY_CHARS = 3500

# Unanswerable questions are requested in batches this size. Asking for all 20
# in one reply overruns max_tokens and truncates the array.
UNANSWERABLE_BATCH = 5

# Batches keep going until enough *distinct* questions accumulate; this caps the
# retries so a model that keeps repeating itself cannot loop forever.
UNANSWERABLE_MAX_ATTEMPTS = 12

# Token-overlap at or above which two questions are treated as the same probe.
NEAR_DUPLICATE_THRESHOLD = 0.5

_DUPLICATE_STOPWORDS = frozenset(
    "the a an of in on for to with and or was were is are did does how what which when "
    "why many much used using use during after before their its this that these those".split()
)

# Generation is lossy: the leakage filter rejects some questions, and the model
# declines pairs it cannot honestly build a question from. Measured over two
# full runs — single-hop loses ~25% to leakage, multi-hop declines ~20%,
# comparative declines ~53% — so each type is oversampled to land on target.
#
# Comparative is supply-constrained rather than prompt-constrained: the corpus
# holds only 30 same-topic pairings, so oversampling past that just requests
# pairs that do not exist. `sample_pairs` caps at what is available, and the
# target itself was lowered to match.
OVERSAMPLE = {"single_hop": 1.6, "multi_hop": 1.4, "comparative": 2.5}


def load_prompt(name: str) -> str:
    """Read a prompt template.

    :param name: Template stem, e.g. ``"single_hop"``.
    :returns: Template text.
    :raises FileNotFoundError: If the template is missing.
    """
    path = PROMPTS_DIR / f"{name}_{eval_config.PROMPT_VERSION}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def is_near_duplicate(candidate: str, existing: Sequence[str], threshold: float = NEAR_DUPLICATE_THRESHOLD) -> bool:
    """Whether a question repeats one already kept, allowing for surface edits.

    Exact-match dedup is not enough here. Asked repeatedly at temperature 0,
    the model returns the same question with a number swapped — "the
    70B-parameter variant" and "the 1.2T-parameter variant" on the same
    benchmark are one probe, not two. Token-set overlap catches that; digits
    are dropped so a changed figure cannot disguise a repeat.

    :param candidate: Question being considered.
    :param existing: Questions already kept.
    :param threshold: Jaccard overlap at or above which they count as the same.
    :returns: True if the candidate repeats an existing question.
    """

    def content_tokens(text: str) -> Set[str]:
        tokens = re.findall(r"[a-z]+", text.lower())
        return {token for token in tokens if token not in _DUPLICATE_STOPWORDS and len(token) > 2}

    candidate_tokens = content_tokens(candidate)
    if not candidate_tokens:
        return True

    for other in existing:
        other_tokens = content_tokens(other)
        if not other_tokens:
            continue
        overlap = len(candidate_tokens & other_tokens) / len(candidate_tokens | other_tokens)
        if overlap >= threshold:
            return True

    return False


def salvage_objects(text: str) -> List[dict]:
    """Extract every complete JSON object from a possibly truncated array.

    A batch request that runs into ``max_tokens`` returns a valid prefix of an
    array and then stops mid-object. Discarding the whole reply would throw
    away the questions that *did* complete, so the complete ones are recovered
    by brace matching.

    :param text: Raw model output.
    :returns: The objects that parsed, in order.
    """
    objects: List[dict] = []
    depth = 0
    start = None
    in_string = False
    escaped = False

    for position, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = position
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(text[start : position + 1]))
                except json.JSONDecodeError:
                    pass
                start = None

    return objects


def parse_json_response(text: str) -> Any:
    """Parse the model's JSON reply, tolerating fences and truncation.

    Qwen emits fence-free JSON at temperature 0 in practice, but two failure
    modes still have to be handled: a fenced reply, and a batch reply cut off
    by ``max_tokens`` mid-array.

    :param text: Raw model output.
    :returns: The parsed object or array.
    :raises ValueError: If nothing usable can be recovered.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Order matters. Matching `\{.*\}` first against an *array* of objects spans
    # from the first object's brace to the last one's, which parses as "Extra
    # data" rather than as the array. Try the array shape first.
    leading = cleaned.lstrip()[:1]
    patterns = (r"\[.*\]", r"\{.*\}") if leading == "[" else (r"\{.*\}", r"\[.*\]")

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    # Truncated array: keep whatever objects completed.
    salvaged = salvage_objects(cleaned)
    if salvaged:
        logger.warning(f"Model output was truncated; salvaged {len(salvaged)} complete object(s)")
        return salvaged if leading == "[" else salvaged[0]

    raise ValueError(f"No JSON found in model output: {text[:200]!r}")


async def call_model(llm, prompt: str) -> Any:
    """Send a prompt to the generator and parse its JSON reply.

    :param llm: ``BedrockClient`` instance.
    :param prompt: Fully rendered prompt.
    :returns: Parsed JSON.
    """
    result = await llm.generate(
        model=eval_config.GENERATOR_MODEL,
        prompt=prompt,
        max_tokens=eval_config.GENERATION_MAX_TOKENS,
        **eval_config.GENERATION_PARAMS,
    )
    return parse_json_response(result["response"])


def truncate(body: str) -> str:
    """Trim an excerpt to the prompt budget on a sentence boundary where possible.

    :param body: Chunk body.
    :returns: Trimmed body.
    """
    if len(body) <= MAX_BODY_CHARS:
        return body
    cut = body[:MAX_BODY_CHARS]
    last_stop = cut.rfind(". ")
    return cut[: last_stop + 1] if last_stop > MAX_BODY_CHARS // 2 else cut


def make_relevant_chunk(chunk: Chunk, grade: int, snippet: str = "") -> RelevantChunk:
    """Build a ground-truth entry for a chunk.

    :param chunk: The chunk.
    :param grade: 2 for the answer chunk, 1 for supporting context.
    :param snippet: Verbatim answer sentence, used to re-anchor after re-chunking.
    :returns: The ground-truth entry.
    """
    return RelevantChunk(
        arxiv_id=chunk.arxiv_id,
        chunk_index=chunk.chunk_index,
        grade=grade,
        chunk_text=chunk.body,
        section_title=chunk.section_title,
        text_sha256=text_sha256(chunk.full_text),
        answer_snippet=snippet,
    )


def neighbour_of(chunk: Chunk, by_paper: Dict[str, List[Chunk]]) -> Optional[Chunk]:
    """Find the adjacent chunk, which shares 100 words of overlap with this one.

    The overlap means the next chunk usually carries part of the same claim, so
    retrieving it is partial credit rather than a miss. Graded 1.

    :param chunk: The answer chunk.
    :param by_paper: Chunks grouped by paper.
    :returns: The following chunk, or None.
    """
    siblings = by_paper.get(chunk.arxiv_id, [])
    for position, candidate in enumerate(siblings):
        if candidate.chunk_index == chunk.chunk_index:
            return siblings[position + 1] if position + 1 < len(siblings) else None
    return None


def generator_stamp() -> Dict[str, Any]:
    """Record which model and prompt produced a row.

    :returns: Provenance fields.
    """
    return {
        "model": eval_config.GENERATOR_MODEL,
        "prompt_version": eval_config.PROMPT_VERSION,
        **eval_config.GENERATION_PARAMS,
    }


async def generate_single_hop(
    llm,
    seeds: Sequence[Chunk],
    by_paper: Dict[str, List[Chunk]],
    topic_by_paper: Dict[str, str],
    topic_labels: Dict[str, str],
    start_index: int,
) -> List[GoldenQuestion]:
    """Generate one question per seed chunk.

    :param llm: Bedrock client.
    :param seeds: Seed chunks.
    :param by_paper: All eligible chunks grouped by paper.
    :param topic_by_paper: ``arxiv_id`` to topic key.
    :param topic_labels: Topic key to human-readable label.
    :param start_index: First qid number.
    :returns: Generated questions.
    """
    template = load_prompt("single_hop")
    questions: List[GoldenQuestion] = []

    for offset, chunk in enumerate(seeds):
        topic_key = topic_by_paper.get(chunk.arxiv_id, "")
        prompt = (
            template.replace("{topic_label}", topic_labels.get(topic_key, "machine learning"))
            .replace("{section_title}", chunk.section_title or "body")
            .replace("{chunk_body}", truncate(chunk.body))
        )

        try:
            payload = await call_model(llm, prompt)
        except Exception as exc:
            logger.error(f"  single_hop seed {chunk.key} failed: {exc}")
            continue

        if not payload.get("question"):
            logger.info(f"  single_hop seed {chunk.key} declined: {payload.get('reason', 'no reason given')}")
            continue

        relevant = [make_relevant_chunk(chunk, GRADE_ANSWER, payload.get("answer_snippet", ""))]
        neighbour = neighbour_of(chunk, by_paper)
        if neighbour is not None:
            relevant.append(make_relevant_chunk(neighbour, GRADE_SUPPORTING))

        questions.append(
            GoldenQuestion(
                qid=f"q{start_index + offset:04d}",
                question=payload["question"].strip(),
                question_type="single_hop",
                answerable=True,
                relevant_chunks=relevant,
                relevant_papers=sorted({c.arxiv_id for c in relevant}),
                reference_answer=payload.get("reference_answer", "").strip(),
                must_mention=[str(m) for m in payload.get("must_mention", [])][:4],
                generated_from={
                    "arxiv_id": chunk.arxiv_id,
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title,
                },
                topic_key=topic_key,
                generator=generator_stamp(),
            )
        )
        logger.info(f"  [{len(questions)}] {chunk.arxiv_id} c{chunk.chunk_index}: {payload['question'][:78]}")

    return questions


async def generate_pairs(
    llm,
    pairs: Sequence[PairSeed],
    question_type: str,
    topic_by_paper: Dict[str, str],
    start_index: int,
) -> List[GoldenQuestion]:
    """Generate multi-hop or comparative questions from chunk pairs.

    :param llm: Bedrock client.
    :param pairs: Seed pairs.
    :param question_type: ``"multi_hop"`` or ``"comparative"``.
    :param topic_by_paper: ``arxiv_id`` to topic key.
    :param start_index: First qid number.
    :returns: Generated questions.
    """
    template = load_prompt(question_type)
    questions: List[GoldenQuestion] = []

    for offset, pair in enumerate(pairs):
        prompt = (
            template.replace("{section_a}", pair.first.section_title or "body")
            .replace("{body_a}", truncate(pair.first.body))
            .replace("{section_b}", pair.second.section_title or "body")
            .replace("{body_b}", truncate(pair.second.body))
        )

        try:
            payload = await call_model(llm, prompt)
        except Exception as exc:
            logger.error(f"  {question_type} pair {pair.first.key}+{pair.second.key} failed: {exc}")
            continue

        if not payload.get("question"):
            logger.info(f"  {question_type} pair declined: {payload.get('reason', 'no reason given')}")
            continue

        relevant = [
            make_relevant_chunk(pair.first, GRADE_ANSWER, payload.get("snippet_a", "")),
            make_relevant_chunk(pair.second, GRADE_ANSWER, payload.get("snippet_b", "")),
        ]

        questions.append(
            GoldenQuestion(
                qid=f"q{start_index + offset:04d}",
                question=payload["question"].strip(),
                question_type=question_type,
                answerable=True,
                relevant_chunks=relevant,
                relevant_papers=sorted({c.arxiv_id for c in relevant}),
                reference_answer=payload.get("reference_answer", "").strip(),
                must_mention=[str(m) for m in payload.get("must_mention", [])][:4],
                generated_from={
                    "pair": [
                        {"arxiv_id": pair.first.arxiv_id, "chunk_index": pair.first.chunk_index},
                        {"arxiv_id": pair.second.arxiv_id, "chunk_index": pair.second.chunk_index},
                    ],
                    "shared_topic": pair.shared_topic,
                },
                topic_key=pair.shared_topic or topic_by_paper.get(pair.first.arxiv_id, ""),
                generator=generator_stamp(),
            )
        )
        logger.info(f"  [{len(questions)}] {question_type}: {payload['question'][:78]}")

    return questions


async def generate_unanswerable(llm, topic_labels: Sequence[str], count: int, start_index: int) -> List[GoldenQuestion]:
    """Generate negative-control questions the corpus cannot answer.

    :param llm: Bedrock client.
    :param topic_labels: Human-readable corpus topics, so the model knows what
        to avoid.
    :param count: How many to request.
    :param start_index: First qid number.
    :returns: Generated questions with empty ground truth.
    """
    template = load_prompt("unanswerable")
    topics = "\n".join(f"- {label}" for label in topic_labels)
    kept: List[str] = []

    # Requested in small batches. Asking for 20 at once overruns max_tokens and
    # the reply is truncated mid-array — which is how a whole run produced zero
    # negative controls. Batching also lets one bad reply cost 5 questions
    # rather than all of them.
    entries: List[dict] = []
    attempts = 0

    while len(entries) < count and attempts < UNANSWERABLE_MAX_ATTEMPTS:
        attempts += 1
        batch_size = min(UNANSWERABLE_BATCH, count - len(entries))

        # Feeding back what has already been kept is what makes each batch
        # different. At temperature 0 an unchanged prompt returns the same
        # questions every time, and near-identical variants ("70B" vs "1.2T"
        # on the same benchmark) slip past exact-match dedup — a first run
        # produced 20 questions that collapsed to 12 distinct probes.
        avoid = "\n".join(f"- {text}" for text in kept) if kept else "(nothing yet)"
        prompt = template.replace("{topic_list}", topics).replace("{count}", str(batch_size)).replace("{avoid_list}", avoid)

        try:
            payload = await call_model(llm, prompt)
        except Exception as exc:
            logger.error(f"  unanswerable batch failed: {exc}")
            break

        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            logger.error(f"  expected a JSON array, got {type(payload).__name__}")
            break

        added = 0
        for entry in payload:
            text = (entry.get("question") or "").strip()
            if not text or is_near_duplicate(text, kept):
                continue
            kept.append(text)
            entries.append(entry)
            added += 1

        logger.info(f"  batch {attempts}: +{added} distinct ({len(entries)}/{count})")

    if len(entries) < count:
        logger.warning(
            f"  only {len(entries)} of {count} distinct unanswerable questions after {attempts} batches; "
            "the model is running out of genuinely different out-of-scope subjects"
        )

    questions: List[GoldenQuestion] = []
    for offset, entry in enumerate(entries[:count]):
        questions.append(
            GoldenQuestion(
                qid=f"q{start_index + offset:04d}",
                question=entry["question"].strip(),
                question_type="unanswerable",
                answerable=False,
                relevant_chunks=[],
                relevant_papers=[],
                reference_answer="",
                must_mention=[],
                generated_from={"kind": entry.get("kind", ""), "why": entry.get("why_unanswerable", "")},
                generator=generator_stamp(),
            )
        )
        logger.info(f"  [{len(questions)}] unanswerable ({entry.get('kind', '?')}): {entry['question'][:70]}")

    return questions


async def generate_paraphrases(llm, parents: Sequence[GoldenQuestion], start_index: int) -> List[GoldenQuestion]:
    """Rewrite accepted questions in different words, keeping the same ground truth.

    :param llm: Bedrock client.
    :param parents: Questions to paraphrase.
    :param start_index: First qid number.
    :returns: Paraphrase rows linked to their parents.
    """
    template = load_prompt("paraphrase")
    questions: List[GoldenQuestion] = []

    for offset, parent in enumerate(parents):
        prompt = template.replace("{original_question}", parent.question)

        try:
            payload = await call_model(llm, prompt)
        except Exception as exc:
            logger.error(f"  paraphrase of {parent.qid} failed: {exc}")
            continue

        text = (payload.get("question") or "").strip()
        if not text:
            continue

        questions.append(
            GoldenQuestion(
                qid=f"q{start_index + offset:04d}",
                question=text,
                question_type="paraphrase",
                answerable=parent.answerable,
                relevant_chunks=list(parent.relevant_chunks),
                relevant_papers=list(parent.relevant_papers),
                reference_answer=parent.reference_answer,
                must_mention=list(parent.must_mention),
                generated_from=dict(parent.generated_from),
                paraphrase_of=parent.qid,
                topic_key=parent.topic_key,
                generator=generator_stamp(),
            )
        )
        logger.info(f"  [{len(questions)}] paraphrase of {parent.qid}: {text[:70]}")

    return questions


def apply_leakage_filter(questions: List[GoldenQuestion], hapax: Set[str]) -> List[GoldenQuestion]:
    """Reject questions that echo their source chunk's wording.

    Rejected rows are kept with ``review.status = "rejected"`` so the filter's
    decisions stay auditable rather than silently disappearing.

    :param questions: Generated questions.
    :param hapax: Corpus-unique tokens.
    :returns: All questions, with leaky ones marked rejected.
    """
    verdicts = {}

    for question in questions:
        if not question.relevant_chunks:
            continue  # Unanswerable rows have no source to leak from.

        source = " ".join(chunk.chunk_text for chunk in question.relevant_chunks)
        verdict = leak.check_leakage(question.question, source, hapax, eval_config.MAX_TRIGRAM_LEAKAGE)
        verdicts[question.qid] = verdict

        if verdict.leaks:
            question.review = {
                "status": "rejected",
                "reviewer": "leakage_filter",
                "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "notes": verdict.reason,
            }

    for line in leak.summarise(verdicts):
        logger.info(line)

    return questions


async def generate(limit: Optional[int], types: Sequence[str], output: Path) -> int:
    """Run generation end to end.

    :param limit: Cap each question type at N, for smoke tests.
    :param types: Which question types to generate.
    :param output: Draft destination.
    :returns: Process exit code.
    """
    manifest = CorpusManifest.load(eval_config.CORPUS_MANIFEST)
    topic_by_paper = {paper.arxiv_id: paper.topic_key for paper in manifest.papers}
    topic_labels = {topic.key: topic.label for topic in ALL_TOPICS}

    all_chunks = load_chunks()
    if not all_chunks:
        logger.error(f"No chunks in {eval_config.EVAL_INDEX_NAME}. Build the corpus first.")
        return 2

    usable = eligible_chunks(all_chunks)
    by_paper = group_by_paper(usable)
    papers = sorted(by_paper)
    logger.info(f"{len(usable)} eligible chunks across {len(papers)} papers")

    def capped(target: int, question_type: str = "") -> int:
        """Seeds to request, oversampled to survive declines and leakage rejections."""
        if limit:
            return min(target, limit)
        return int(round(target * OVERSAMPLE.get(question_type, 1.0)))

    llm = make_bedrock_client()
    questions: List[GoldenQuestion] = []
    used_keys: Set[tuple] = set()
    next_id = 1

    if "single_hop" in types:
        logger.info("--- single-hop ---")
        seeds = sample_single_hop(usable, capped(eval_config.TARGET_SINGLE_HOP, "single_hop"))
        used_keys.update(chunk.key for chunk in seeds)
        batch = await generate_single_hop(llm, seeds, by_paper, topic_by_paper, topic_labels, next_id)
        questions.extend(batch)
        next_id += len(seeds)

    pair_seeds: List[PairSeed] = []
    for pair_type, target in (("multi_hop", eval_config.TARGET_MULTI_HOP), ("comparative", eval_config.TARGET_COMPARATIVE)):
        if pair_type not in types:
            continue
        logger.info(f"--- {pair_type} ---")
        pairs = sample_pairs(usable, topic_by_paper, capped(target, pair_type), used_keys)
        pair_seeds.extend(pairs)
        batch = await generate_pairs(llm, pairs, pair_type, topic_by_paper, next_id)
        questions.extend(batch)
        next_id += len(pairs)

    if "unanswerable" in types:
        logger.info("--- unanswerable ---")
        labels = [topic_labels[key] for key in sorted({topic_by_paper[p] for p in papers if p in topic_by_paper})]
        batch = await generate_unanswerable(llm, labels, capped(eval_config.TARGET_UNANSWERABLE, "unanswerable"), next_id)
        questions.extend(batch)
        next_id += len(batch)

    logger.info("--- leakage filter ---")
    hapax = leak.build_hapax_vocabulary(chunk.body for chunk in all_chunks)
    questions = apply_leakage_filter(questions, hapax)

    if "paraphrase" in types:
        logger.info("--- paraphrase pairs ---")
        clean = [q for q in questions if q.question_type == "single_hop" and q.review["status"] == "pending"]
        parents = clean[: capped(eval_config.TARGET_PARAPHRASE, "paraphrase")]
        batch = await generate_paraphrases(llm, parents, next_id)
        questions.extend(batch)

    problems = validate_dataset(questions)
    for problem in problems[:20]:
        logger.error(f"schema problem: {problem}")
    if problems:
        logger.error(f"{len(problems)} schema problems found")

    write_jsonl(questions, output)

    counts: Dict[str, int] = {}
    for question in questions:
        counts[question.question_type] = counts.get(question.question_type, 0) + 1
    rejected = sum(1 for q in questions if q.review["status"] == "rejected")

    # Coverage must be measured over *surviving* rows. Counting rejected ones
    # too reports 43/43 while a paper whose only questions all failed the
    # leakage filter has none left — it then sits in the index as a pure
    # distractor, adding retrieval difficulty without ever being a target.
    surviving = [question for question in questions if question.review["status"] != "rejected"]
    covered = {paper for question in surviving for paper in question.relevant_papers}
    uncovered = sorted(set(papers) - covered)

    logger.info("=== Generation summary ===")
    logger.info(f"  questions written : {len(questions)} -> {output}")
    logger.info(f"  surviving         : {len(surviving)} (target {eval_config.TARGET_QUESTIONS})")
    logger.info(f"  by type           : {counts}")
    logger.info(f"  rejected (leakage): {rejected}")
    logger.info(f"  papers covered    : {len(covered)}/{len(papers)} (surviving rows only)")

    if uncovered:
        logger.warning(f"  PAPERS WITH NO SURVIVING QUESTION: {uncovered}")
        logger.warning("  Recover them during review: their leakage-rejected questions are in the draft,")
        logger.warning("  and editing one to remove the copied phrasing restores coverage.")
    logger.info(f"\nNext: python -m src.evaluation.dataset.review --input {output}")

    return 0 if not problems else 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.dataset.generate",
        description="Generate the golden question set from the evaluation index.",
    )
    parser.add_argument("--output", type=Path, default=eval_config.GOLDEN_DRAFT, help="Draft JSONL destination")
    parser.add_argument("--limit", type=int, default=None, help="Cap each question type at N (smoke test)")
    parser.add_argument(
        "--types",
        nargs="+",
        default=["single_hop", "multi_hop", "comparative", "unanswerable", "paraphrase"],
        help="Question types to generate",
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
    setup_logging(eval_config.log_path(run_id, "dataset_generate"), verbose=args.verbose)

    try:
        return asyncio.run(generate(limit=args.limit, types=args.types, output=args.output))
    except KeyboardInterrupt:
        logger.warning("Interrupted before the draft was written")
        return 130
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    except Exception as exc:
        logger.exception(f"Generation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
