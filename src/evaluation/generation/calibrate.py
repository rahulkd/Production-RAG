"""Judge-agreement check: Cohen's kappa between the judge and a human.

Usage::

    python -m src.evaluation.generation.calibrate sample --run 20261023-101500 [--cutoff hybrid_k10]
    #   fill in "human_label" in results/generation_<run>/<cutoff>/calibration_items.jsonl
    python -m src.evaluation.generation.calibrate score --run 20261023-101500 [--cutoff hybrid_k10]

``sample`` draws judged items from a run — claim verdicts, relevance scores and
refusal labels — and writes them **without** the judge's label, so the human
labels blind. The judge labels go to a separate key file. ``score`` joins the
two and writes ``results/judge_calibration.md``.

If kappa is below 0.6 for a rubric, that rubric is broken and every number
built on it is decoration: fix the rubric, bump ``JUDGE_PROMPT_VERSION``, re-run.
"""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Sequence

import src.evaluation._bootstrap  # noqa: F401
from src.evaluation import config as eval_config
from src.evaluation.generation.judge import FAITHFULNESS_LABELS, REFUSAL_LABELS, RELEVANCE_SCORES

KAPPA_THRESHOLD = 0.6

# 40 items in total, weighted towards claim verdicts: faithfulness is the
# headline number and the hardest call the judge makes.
DEFAULT_COUNTS = {"claim": 20, "relevance": 10, "refusal": 10}

ALLOWED_LABELS = {
    "claim": FAITHFULNESS_LABELS,
    "relevance": tuple(str(score) for score in RELEVANCE_SCORES),
    "refusal": REFUSAL_LABELS,
}


def cohen_kappa(first: Sequence[Hashable], second: Sequence[Hashable]) -> float:
    """Cohen's kappa for two raters over the same items.

    :param first: Labels from rater one.
    :param second: Labels from rater two, aligned with ``first``.
    :returns: Kappa in [-1, 1]; 1.0 when both raters use one identical label throughout.
    :raises ValueError: On empty or misaligned input.
    """
    if len(first) != len(second) or not first:
        raise ValueError(f"need two equal-length non-empty label lists, got {len(first)} and {len(second)}")

    n = len(first)
    observed = sum(1 for a, b in zip(first, second) if a == b) / n
    counts_first, counts_second = Counter(first), Counter(second)
    expected = sum(counts_first[label] * counts_second[label] for label in set(first) | set(second)) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def run_dir(run_id: str, cutoff: str) -> Path:
    """Output directory of one cutoff of a generation run.

    :param run_id: Run identifier.
    :param cutoff: Sub-directory such as ``hybrid_k10``.
    :returns: ``results/generation_<run_id>/<cutoff>``.
    """
    return eval_config.RESULTS_DIR / f"generation_{run_id}" / cutoff


def build_items(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Every judged decision in a run, grouped by rubric.

    :param rows: Raw generation rows.
    :returns: Rubric to items; each item carries what a human needs to label it
        and, under ``judge_label``, what the judge said.
    """
    items: Dict[str, List[Dict[str, Any]]] = {kind: [] for kind in DEFAULT_COUNTS}
    for row in rows:
        if "generation_error" in row:
            continue
        if row.get("refusal"):
            items["refusal"].append(
                {"qid": row["qid"], "question": row["question"], "answer": row["answer"], "judge_label": row["refusal"]["label"]}
            )
        # Refusals get relevance 0 by rule, not by the judge, so they are not a judge decision to calibrate.
        if row.get("relevance") and not (row.get("refusal") and row["refusal"]["label"] == "refused"):
            items["relevance"].append(
                {"qid": row["qid"], "question": row["question"], "answer": row["answer"], "judge_label": str(row["relevance"]["score"])}
            )
        for position, verdict in enumerate(row.get("claims") or []):
            items["claim"].append(
                {
                    "qid": row["qid"],
                    "claim_index": position,
                    "claim": verdict["claim"],
                    "context": row["context"],
                    "judge_label": verdict["label"],
                }
            )
    return items


def sample(run_id: str, cutoff: str, counts: Dict[str, int], seed: int) -> Path:
    """Write blind labelling items and their answer key.

    :param run_id: Generation run to sample from.
    :param cutoff: Cutoff sub-directory, e.g. ``hybrid_k10``.
    :param counts: Items per rubric.
    :param seed: RNG seed, so the sample is reproducible.
    :returns: Path of the items file.
    """
    directory = run_dir(run_id, cutoff)
    rows = [json.loads(line) for line in (directory / "generation_raw.jsonl").read_text(encoding="utf-8").splitlines() if line]
    pools = build_items(rows)

    rng = random.Random(seed)
    items, key = [], {}
    for kind, count in counts.items():
        for item in rng.sample(pools[kind], min(count, len(pools[kind]))):
            item_id = f"{kind}-{len(items):03d}"
            key[item_id] = item.pop("judge_label")
            items.append(
                {
                    "item_id": item_id,
                    "kind": kind,
                    "allowed_labels": list(ALLOWED_LABELS[kind]),
                    **item,
                    "human_label": "",
                }
            )

    items_path = directory / "calibration_items.jsonl"
    items_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
    (directory / "calibration_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} items to {items_path}; set each human_label to one of its allowed_labels")
    return items_path


def score(run_id: str, cutoff: str, output: Path) -> Dict[str, Optional[float]]:
    """Compute per-rubric kappa and write the calibration report.

    :param run_id: Generation run that was sampled.
    :param cutoff: Cutoff sub-directory that was sampled.
    :param output: Report destination.
    :returns: Rubric to kappa (None when no labelled items).
    :raises ValueError: If any item is unlabelled or has an invalid label.
    """
    directory = run_dir(run_id, cutoff)
    items = [json.loads(line) for line in (directory / "calibration_items.jsonl").read_text(encoding="utf-8").splitlines() if line]
    key = json.loads((directory / "calibration_key.json").read_text(encoding="utf-8"))

    problems = [
        f"{item['item_id']}: label {item['human_label']!r} not in {item['allowed_labels']}"
        for item in items
        if str(item["human_label"]) not in item["allowed_labels"]
    ]
    if problems:
        raise ValueError(f"{len(problems)} items unlabelled or invalid, e.g. {problems[:3]}")

    results: Dict[str, Optional[float]] = {}
    lines = [
        "# Judge calibration",
        "",
        f"- Generation run: `{run_id}/{cutoff}` · scored {datetime.now(timezone.utc):%Y-%m-%d}",
        f"- Judge: see `generation_{run_id}/{cutoff}/generation_summary.json` · threshold κ ≥ {KAPPA_THRESHOLD}",
        "",
        "| rubric | items | agreement | Cohen's κ | verdict |",
        "|---|---|---|---|---|",
    ]
    disagreements = []
    for kind in DEFAULT_COUNTS:
        subset = [item for item in items if item["kind"] == kind]
        if not subset:
            results[kind] = None
            continue
        human = [str(item["human_label"]) for item in subset]
        judge = [key[item["item_id"]] for item in subset]
        kappa = cohen_kappa(human, judge)
        results[kind] = kappa
        agreement = sum(1 for a, b in zip(human, judge) if a == b) / len(subset)
        verdict = "ok" if kappa >= KAPPA_THRESHOLD else "**rubric broken — fix before trusting scores**"
        lines.append(f"| {kind} | {len(subset)} | {agreement:.0%} | {kappa:.2f} | {verdict} |")
        disagreements += [(item, j) for item, h, j in zip(subset, human, judge) if h != j]

    if disagreements:
        lines += ["", "## Disagreements", "", "| item | qid | human | judge |", "|---|---|---|---|"]
        lines += [f"| {item['item_id']} | {item['qid']} | {item['human_label']} | {judge} |" for item, judge in disagreements]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for kind, kappa in results.items():
        print(f"  {kind:<10} κ={'–' if kappa is None else f'{kappa:.2f}'}")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Process exit code; 1 if any rubric is below threshold.
    """
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.generation.calibrate", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sample_parser = sub.add_parser("sample", help="Write blind labelling items for a run")
    sample_parser.add_argument("--run", required=True, help="Generation run id")
    sample_parser.add_argument("--cutoff", default="hybrid_k10", help="Cutoff sub-directory (default hybrid_k10)")
    sample_parser.add_argument("--seed", type=int, default=13)
    for kind, count in DEFAULT_COUNTS.items():
        sample_parser.add_argument(f"--{kind}", type=int, default=count, help=f"{kind} items (default {count})")

    score_parser = sub.add_parser("score", help="Compute kappa from the labelled items")
    score_parser.add_argument("--run", required=True, help="Generation run id")
    score_parser.add_argument("--cutoff", default="hybrid_k10", help="Cutoff sub-directory (default hybrid_k10)")
    score_parser.add_argument("--output", type=Path, default=eval_config.JUDGE_CALIBRATION)

    args = parser.parse_args(argv)
    try:
        if args.command == "sample":
            sample(args.run, args.cutoff, {kind: getattr(args, kind) for kind in DEFAULT_COUNTS}, args.seed)
            return 0
        kappas = score(args.run, args.cutoff, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if all(kappa is None or kappa >= KAPPA_THRESHOLD for kappa in kappas.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
