"""Markdown rendering for evaluation results.

The per-run report is overwritten with each run; ``leaderboard.md`` is only
ever appended to. The leaderboard is what turns individual measurements into a
history that a config change can be diffed against.
"""

from pathlib import Path
from typing import Any, Dict, List

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
