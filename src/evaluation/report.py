"""Markdown rendering for evaluation results.

The per-run report is overwritten with each run; ``leaderboard.md`` is only
ever appended to. The leaderboard is what turns individual measurements into a
history that a config change can be diffed against.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.retrieval.metrics import METRIC_NAMES

METRIC_LABELS = {"recall": "R@k", "ndcg": "nDCG@k", "mrr": "MRR@k", "hit": "Hit@k", "paper_recall": "paperR@k"}

LEADERBOARD_HEADER = [
    "# Retrieval leaderboard",
    "",
    "Appended by `python -m src.evaluation.retrieval.runner`; never edit or reorder rows.",
    "Every metric is taken at the row's own `k`. Values are mean ± 95% CI over the answerable golden questions.",
    "Recall, MRR and Hit count answer chunks (grade 2) only; nDCG also credits the neighbour chunk (grade 1).",
    "Latency is the OpenSearch call alone; vector and hybrid exclude the query-embedding call.",
    "",
    "| run_id | mode | k | R@k | nDCG@k | MRR@k | Hit@k | paperR@k | n | p50 ms | p95 ms | dataset |",
    "|--------|------|---|-----|--------|-------|-------|----------|---|--------|--------|---------|",
]


def fmt(stat: Dict[str, float], with_ci: bool = True) -> str:
    """Format a :func:`mean_ci` result.

    :param stat: ``{"mean", "ci95", "n"}``.
    :param with_ci: Append the half-width.
    :returns: e.g. ``"0.712 ±.051"``.
    """
    if stat["n"] == 0:
        return "–"
    mean = f"{stat['mean']:.3f}"
    return f"{mean} ±{stat['ci95']:.3f}".replace("±0.", "±.") if with_ci else mean


def leaderboard_rows(result: Dict[str, Any]) -> List[str]:
    """One leaderboard row per config in a run.

    :param result: Run summary as written to ``retrieval_summary.json``.
    :returns: Markdown table rows.
    """
    rows = []
    for summary in result["configs"]:
        overall = summary["overall"]
        latency = summary["latency_ms"]
        cells = [
            result["run_id"],
            summary["mode"],
            str(summary["top_k"]),
            *(fmt(overall[name]) for name in METRIC_NAMES),
            str(overall["recall"]["n"]),
            f"{latency['p50']:.0f}",
            f"{latency['p95']:.0f}",
            result["dataset_sha256"][:8],
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def append_leaderboard(path: Path, result: Dict[str, Any]) -> None:
    """Append a run's rows to the leaderboard, creating it with a header if absent.

    :param path: ``leaderboard.md``.
    :param result: Run summary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [] if path.exists() else list(LEADERBOARD_HEADER)
    lines.extend(leaderboard_rows(result))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def render_retrieval_report(result: Dict[str, Any]) -> str:
    """Render one retrieval run as Markdown.

    :param result: Run summary.
    :returns: The report.
    """
    configs = result["configs"]
    lines = [
        f"# Retrieval run {result['run_id']}",
        "",
        f"- Dataset: `{result['dataset']}` (sha256 `{result['dataset_sha256'][:12]}`)",
        f"- Index: `{result['index']}` ({result['index_doc_count']} chunks)",
        f"- Questions: {result['questions']} answerable — {result['question_types']}",
        "",
        "Metrics are at each config's own k, mean ± 95% CI. Recall, MRR and Hit count answer chunks (grade 2);",
        "nDCG also credits the neighbour chunk (grade 1). Latency is the search call only, excluding query embedding.",
        "",
        "## Overall",
        "",
        "| config | " + " | ".join(METRIC_LABELS[name] for name in METRIC_NAMES) + " | p50 ms | p95 ms |",
        "|---" * (len(METRIC_NAMES) + 3) + "|",
    ]
    for summary in configs:
        overall = summary["overall"]
        latency = summary["latency_ms"]
        lines.append(
            f"| {summary['config']} | "
            + " | ".join(fmt(overall[name]) for name in METRIC_NAMES)
            + f" | {latency['p50']:.0f} | {latency['p95']:.0f} |"
        )

    # Q2 reads off this directly: how much recall each mode gains as k grows.
    modes = list(dict.fromkeys(summary["mode"] for summary in configs))
    top_ks = sorted({summary["top_k"] for summary in configs})
    by_mode_k = {(summary["mode"], summary["top_k"]): summary for summary in configs}

    for name, title in (("recall", "Recall@k by mode"), ("ndcg", "nDCG@k by mode")):
        lines += ["", f"## {title}", "", "| mode | " + " | ".join(f"k={k}" for k in top_ks) + " |", "|---" * (len(top_ks) + 1) + "|"]
        for mode in modes:
            cells = [fmt(by_mode_k[(mode, k)]["overall"][name], with_ci=False) if (mode, k) in by_mode_k else "–" for k in top_ks]
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")

    lines += ["", "## By question type", ""]
    question_types = sorted({question_type for summary in configs for question_type in summary["by_type"]})
    lines += [
        "| config | " + " | ".join(f"{t} R / nDCG (n)" for t in question_types) + " |",
        "|---" * (len(question_types) + 1) + "|",
    ]
    for summary in configs:
        cells = []
        for question_type in question_types:
            stats = summary["by_type"].get(question_type)
            if stats is None:
                cells.append("–")
                continue
            cells.append(f"{fmt(stats['recall'], False)} / {fmt(stats['ndcg'], False)} ({stats['recall']['n']})")
        lines.append(f"| {summary['config']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Paraphrase gap (Q4)",
        "",
        "Original minus paraphrase, same ground truth. Positive = the reworded question scored lower,",
        "i.e. the retriever leans on surface wording. Few pairs, so read the CI before the mean.",
        "",
        "| config | pairs | Δ Recall | Δ nDCG |",
        "|---|---|---|---|",
    ]
    for summary in configs:
        gap = summary["paraphrase_gap"]
        lines.append(f"| {summary['config']} | {gap['pairs']} | {fmt(gap['recall'])} | {fmt(gap['ndcg'])} |")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Generation (Phase 4)
# --------------------------------------------------------------------------

GENERATION_LEADERBOARD_HEADER = [
    "# Generation leaderboard",
    "",
    "Appended by `python -m src.evaluation.generation.runner`; never edit or reorder rows.",
    "Values are mean ± 95% CI. Faithfulness = supported claims / claims, over attempted answers; contradictions per answer",
    "are reported separately. Relevance is 0-3 over answerable questions (a refusal scores 0). Citation precision = cited",
    "papers that were retrieved / cited papers. Refusal recall is over unanswerable questions, false refusal over answerable.",
    "",
    "| run_id | retrieval | judge | faithful | contra/ans | relevance | cite prec | cite present | refusal recall | false refusal | json ok | n | dataset |",
    "|--------|-----------|-------|----------|------------|-----------|-----------|--------------|----------------|---------------|---------|---|---------|",
]


def generation_leaderboard_row(result: Dict[str, Any]) -> str:
    """One leaderboard row for a generation run.

    :param result: Run summary as written to ``generation_summary.json``.
    :returns: Markdown table row.
    """
    summary = result["summary"]
    faithfulness = summary["faithfulness"]
    cells = [
        result["run_id"],
        result["retrieval"],
        # Drop only the cross-region prefix: "deepseek.v3.2" must not collapse to "2".
        re.sub(r"^(us|eu|apac|global)\.", "", result["judge_model"]),
        fmt(faithfulness["score"]),
        fmt(faithfulness["contradicted_per_answer"], with_ci=False),
        fmt(summary["relevance"]),
        fmt(summary["citations"]["precision"]),
        fmt(summary["citations"]["present_rate"], with_ci=False),
        fmt(summary["refusal"]["recall"]),
        fmt(summary["refusal"]["false_refusal_rate"]),
        fmt(summary["json_ok"], with_ci=False),
        str(summary["rows"] - summary["generation_errors"]),
        result["dataset_sha256"][:8],
    ]
    return "| " + " | ".join(cells) + " |"


def append_generation_leaderboard(path: Path, result: Dict[str, Any]) -> None:
    """Append a generation run to its leaderboard, creating it with a header if absent.

    :param path: ``generation_leaderboard.md``.
    :param result: Run summary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [] if path.exists() else list(GENERATION_LEADERBOARD_HEADER)
    lines.append(generation_leaderboard_row(result))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def render_generation_report(result: Dict[str, Any]) -> str:
    """Render one generation run as Markdown.

    :param result: Run summary.
    :returns: The report.
    """
    summary = result["summary"]
    faithfulness = summary["faithfulness"]
    citations = summary["citations"]
    field = summary["field_citations"]
    refusal = summary["refusal"]
    usage = result["usage"]

    lines = [
        f"# Generation run {result['run_id']}",
        "",
        f"- Dataset: `{result['dataset']}` (sha256 `{result['dataset_sha256'][:12]}`)",
        f"- Index: `{result['index']}` ({result['index_doc_count']} chunks), retrieval `{result['retrieval']}`",
        f"- System under test: `{result['sut_model']}` {result['sut_params']}, `/ask` structured prompt",
        f"- Judge: `{result['judge_model']}` {result['judge_params']}, rubric `{result['judge_prompt_version']}`"
        + ("" if result["judge_cache_read"] else " (cache bypassed)"),
        f"- Questions: {result['questions']} — {result['question_types']}",
        f"- Generation errors: {summary['generation_errors']} · judge errors: {summary['judge_errors']} · "
        f"answers truncated at max_tokens: {summary['truncated']}",
        f"- Judge calls: {usage['judge_calls']} (+{usage['judge_cache_hits']} cached) · tokens in/out: "
        f"judge {usage['judge']['inputTokens']}/{usage['judge']['outputTokens']}, "
        f"generator {usage['generator']['inputTokens']}/{usage['generator']['outputTokens']}",
        "",
        "## Scorecard",
        "",
        "No single scalar: faithfulness and refusal trade off, and a combined number would hide it.",
        "Citation rows without a qualifier are inline `[arXiv:id]` markers in the answer text — what a reader sees.",
        "",
        "| metric | value | n | question |",
        "|---|---|---|---|",
        f"| Faithfulness (supported / claims) | {fmt(faithfulness['score'])} | {faithfulness['score']['n']} answers, {faithfulness['claims']} claims | Q7 |",
        f"| Contradicted claims per answer | {fmt(faithfulness['contradicted_per_answer'])} | {faithfulness['contradicted_per_answer']['n']} | Q7 |",
        f"| Answers with ≥1 contradiction | {fmt(faithfulness['answers_with_contradiction'])} | {faithfulness['answers_with_contradiction']['n']} | Q7 |",
        f"| Answer relevance (0-3) | {fmt(summary['relevance'])} | {summary['relevance']['n']} | — |",
        f"| Answered (relevance ≥ 2) | {fmt(summary['answered_rate'])} | {summary['answered_rate']['n']} | — |",
        f"| must_mention coverage | {fmt(summary['must_mention'])} | {summary['must_mention']['n']} | — |",
        f"| Citation precision | {fmt(citations['precision'])} | {citations['precision']['n']} | Q8 |",
        f"| Citation recall | {fmt(citations['recall'])} | {citations['recall']['n']} | Q8 |",
        f"| Answers with ≥1 citation | {fmt(citations['present_rate'])} | {citations['present_rate']['n']} | Q8 |",
        f"| Hallucinated citations | {citations['hallucinated_total']} in {citations['answers_with_hallucinated']} answers | — | Q8 |",
        f"| JSON `citations` field: precision | {fmt(field['precision'])} | {field['precision']['n']} | Q8 |",
        f"| JSON `citations` field: ≥1 arXiv ID | {fmt(field['present_rate'])} | {field['present_rate']['n']} | Q8 |",
        f"| JSON `citations` field: hallucinated IDs | {field['hallucinated_total']} in {field['answers_with_hallucinated']} answers | — | Q8 |",
        f"| JSON `citations` field: non-arXiv entries (\"[11]\", \"X et al.\") | in {field['answers_with_non_arxiv_entries']} answers | — | Q8 |",
        f"| Refusal recall (unanswerable refused) | {fmt(refusal['recall'])} | {refusal['recall']['n']} | Q9 |",
        f"| False refusal rate (answerable refused) | {fmt(refusal['false_refusal_rate'])} | {refusal['false_refusal_rate']['n']} | Q9 |",
        f"| Valid structured JSON | {fmt(summary['json_ok'])} | {summary['json_ok']['n']} | C6 |",
        "",
        "## Faithfulness split",
        "",
        "Attempted answers to unanswerable questions are the hallucination case: any claim there is unsupported by design.",
        "",
        "| subset | faithful | contra/ans | answers | claims |",
        "|---|---|---|---|---|",
    ]
    for label, key in (("answerable", "faithfulness_answerable"), ("unanswerable, attempted", "faithfulness_unanswerable")):
        block = summary[key]
        lines.append(
            f"| {label} | {fmt(block['score'])} | {fmt(block['contradicted_per_answer'], with_ci=False)} | {block['score']['n']} | {block['claims']} |"
        )

    lines += [
        "",
        "## Retrieval vs generation (Q10)",
        "",
        "Answerable questions split by whether an answer chunk (grade 2) was in the generator's context.",
        "Poor answers *with* the chunk retrieved point at generation; poor answers without it point at retrieval.",
        "",
        "| answer chunk | n | answered (rel ≥ 2) | relevance | faithfulness |",
        "|---|---|---|---|---|",
    ]
    for label, block in summary["bottleneck"].items():
        lines.append(f"| {label} | {block['n']} | {block['answered']} | {fmt(block['relevance'])} | {fmt(block['faithfulness'])} |")

    lines += [
        "",
        "## By question type",
        "",
        "| type | n | faithfulness | relevance | refused |",
        "|---|---|---|---|---|",
    ]
    for question_type, block in summary["by_type"].items():
        lines.append(
            f"| {question_type} | {block['n']} | {fmt(block['faithfulness'], False)} | {fmt(block['relevance'], False)} | {fmt(block['refused'], False)} |"
        )

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# End-to-end (Phase 5)
# --------------------------------------------------------------------------

E2E_LEADERBOARD_HEADER = [
    "# End-to-end leaderboard",
    "",
    "Appended by `python -m src.evaluation.e2e.runner`; never edit or reorder rows.",
    "Each row is the golden set sent over HTTP to one route at one top_k. `agree` = share of responses whose source papers",
    "equal the in-process harness retrieval for the same query (Q11). Quality columns use the Phase 4 judge on the served answer;",
    "faithfulness is scored only where the papers agree. Latency is wall-clock per request; TTFT applies to /stream only.",
    "",
    "| run_id | route | k | agree | hybrid | faithful | relevance | refusal recall | false refusal | cite present | p50 s | p95 s | TTFT p50 s | errors | dataset |",
    "|--------|-------|---|-------|--------|----------|-----------|----------------|---------------|--------------|-------|-------|------------|--------|---------|",
]


def _secs(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:.1f}"


def e2e_leaderboard_row(result: Dict[str, Any]) -> str:
    """One leaderboard row for one route and k.

    :param result: As written to ``e2e_summary.json``.
    :returns: Markdown table row.
    """
    summary = result["summary"]
    quality = summary["quality"]
    served = sum(summary["search_modes"].values())
    cells = [
        result["run_id"],
        f"/{result['endpoint']}",
        str(result["top_k"]),
        fmt(summary["papers_agree"], with_ci=False),
        f"{summary['search_modes'].get('hybrid', 0)}/{served}",
        fmt(quality["faithfulness"]["score"]),
        fmt(quality["relevance"]),
        fmt(quality["refusal"]["recall"], with_ci=False),
        fmt(quality["refusal"]["false_refusal_rate"], with_ci=False),
        fmt(quality["citations"]["present_rate"], with_ci=False),
        _secs(summary["latency_s"]["p50"]),
        _secs(summary["latency_s"]["p95"]),
        _secs(summary["ttft_s"]["p50"]),
        str(summary["http_errors"] + quality["judge_errors"]),
        result["dataset_sha256"][:8],
    ]
    return "| " + " | ".join(cells) + " |"


def append_e2e_leaderboard(path: Path, results: List[Dict[str, Any]]) -> None:
    """Append one row per route and k, creating the file with a header if absent.

    :param path: ``e2e_leaderboard.md``.
    :param results: One result per config.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [] if path.exists() else list(E2E_LEADERBOARD_HEADER)
    lines.extend(e2e_leaderboard_row(result) for result in results)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def render_e2e_report(results: List[Dict[str, Any]]) -> str:
    """Render a Phase 5 run, every route and k side by side.

    :param results: One result per config.
    :returns: The report.
    """
    first = results[0]
    names = [f"/{r['endpoint']} k={r['top_k']}" for r in results]

    def row(label: str, cells: List[str]) -> str:
        return f"| {label} | " + " | ".join(cells) + " |"

    def each(getter) -> List[str]:
        return [getter(r["summary"]) for r in results]

    lines = [
        f"# End-to-end run {first['run_id']}",
        "",
        f"- API: `{first['base_url']}` · generator `{first['sut_model']}` (temperature 0.7, hardcoded in the routes)",
        f"- Dataset: `{first['dataset']}` (sha256 `{first['dataset_sha256'][:12]}`), {first['questions']} questions",
        f"- Reference retrieval: in-process hybrid on `{first['index']}` ({first['index_doc_count']} chunks)",
        f"- Judge: `{first['judge_model']}`, rubric `{first['judge_prompt_version']}`",
        "",
        "`/ask` uses the structured JSON prompt; `/stream` (the only route the UI calls) uses the plain prompt.",
        "",
        "## Does the API behave like the harness? (Q11)",
        "",
        "| metric | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
        row("HTTP / route errors", each(lambda s: str(s["http_errors"]))),
        row("search_mode", each(lambda s: ", ".join(f"{k}: {v}" for k, v in s["search_modes"].items()))),
        row("Sources = harness papers", each(lambda s: fmt(s["papers_agree"]))),
        row("Paper recall via API", each(lambda s: fmt(s["api_paper_recall"]))),
        row("Paper recall in-process", each(lambda s: fmt(s["harness_paper_recall"]))),
        row("\"No relevant information\" replies", each(lambda s: str(s["no_chunk_answers"]))),
        "",
        "## Latency (seconds, wall-clock per request)",
        "",
        "| metric | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
        row("p50", each(lambda s: _secs(s["latency_s"]["p50"]))),
        row("p95", each(lambda s: _secs(s["latency_s"]["p95"]))),
        row("Time to first token p50", each(lambda s: _secs(s["ttft_s"]["p50"]))),
        row("Time to first token p95", each(lambda s: _secs(s["ttft_s"]["p95"]))),
        "",
        "## Answer quality as served",
        "",
        "| metric | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
        row("Faithfulness", each(lambda s: fmt(s["quality"]["faithfulness"]["score"]))),
        row("Answers with ≥1 contradiction", each(lambda s: fmt(s["quality"]["faithfulness"]["answers_with_contradiction"], with_ci=False))),
        row("Relevance (0-3)", each(lambda s: fmt(s["quality"]["relevance"]))),
        row("Answered (relevance ≥ 2)", each(lambda s: fmt(s["quality"]["answered_rate"], with_ci=False))),
        row("Refusal recall", each(lambda s: fmt(s["quality"]["refusal"]["recall"], with_ci=False))),
        row("False refusal rate", each(lambda s: fmt(s["quality"]["refusal"]["false_refusal_rate"], with_ci=False))),
        row("Answers with inline citation", each(lambda s: fmt(s["quality"]["citations"]["present_rate"], with_ci=False))),
        row("Hallucinated citations", each(lambda s: str(s["quality"]["citations"]["hallucinated_total"]))),
        row("Raw JSON shown as answer (/ask)", each(lambda s: "–" if s["quality"]["json_ok"]["n"] == 0 else str(round((1 - s["quality"]["json_ok"]["mean"]) * s["quality"]["json_ok"]["n"])))),
        row("Judge errors", each(lambda s: str(s["quality"]["judge_errors"]))),
        "",
        "## Faithfulness by question type",
        "",
        "| type | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    types = sorted({t for r in results for t in r["summary"]["quality"]["by_type"]})
    for question_type in types:
        lines.append(
            row(
                question_type,
                [fmt(r["summary"]["quality"]["by_type"].get(question_type, {"faithfulness": {"n": 0}})["faithfulness"], with_ci=False) for r in results],
            )
        )
    return "\n".join(lines) + "\n"
