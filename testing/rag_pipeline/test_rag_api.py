# Complete RAG pipeline test against the /ask API, backed by AWS Bedrock.
# - runs one end-to-end question and prints the answer + provenance
# - time-profiles the endpoint across scenarios (top_k, hybrid vs BM25)
# - splits latency into retrieval vs LLM generation using /hybrid-search as the baseline
# - saves a Markdown report to ./results/
import json
import statistics
import time
from datetime import datetime

import os as _os
import requests

# --- Setup ----------------------------------------------------------------
API_BASE = _os.getenv("RAG_API_BASE", "http://localhost:8000/api/v1")
ASK_URL = f"{API_BASE}/ask"  # note: no trailing slash, the route is declared as "/ask"
SEARCH_URL = f"{API_BASE}/hybrid-search/"
HEALTH_URL = f"{API_BASE}/health"

# Bedrock model id (was "llama3.2:1b" when this ran on local Ollama).
MODEL = _os.getenv("BEDROCK_MODEL", "meta.llama3-70b-instruct-v1:0")

REQUEST_TIMEOUT = 120  # Bedrock 70B is slower than a local 1B model
TIMING_RUNS = 3  # each scenario is timed over this many runs (min + mean reported)

RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results")

QUESTION = "explain poisoning of pretrained data using external web sources  "

# (label, top_k, use_hybrid)
SCENARIOS = [
    ("top_k=1, hybrid", 1, True),
    ("top_k=3, hybrid", 3, True),
    ("top_k=5, hybrid", 5, True),
    ("top_k=3, bm25", 3, False),
]

report_lines = []


def emit(line=""):
    """Print to stdout and collect for the Markdown report."""
    print(line)
    report_lines.append(line)


def ask(question, top_k, use_hybrid, model=MODEL):
    """POST /ask and return (elapsed_seconds, response)."""
    payload = {
        "query": question,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "model": model,
    }
    start = time.perf_counter()
    response = requests.post(ASK_URL, json=payload, timeout=REQUEST_TIMEOUT)
    return time.perf_counter() - start, response


def search_only(question, size, use_hybrid):
    """POST /hybrid-search to measure retrieval cost without the LLM."""
    payload = {"query": question, "size": size, "use_hybrid": use_hybrid}
    start = time.perf_counter()
    response = requests.post(SEARCH_URL, json=payload, timeout=REQUEST_TIMEOUT)
    return time.perf_counter() - start, response


def summarize(samples):
    """Reduce a list of latencies to the stats used in the report."""
    return {
        "runs": len(samples),
        "min": min(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "max": max(samples),
    }


# --- Preflight ------------------------------------------------------------
emit("COMPLETE RAG PIPELINE TEST (BEDROCK)")
emit("=" * 40)
emit(f"Endpoint: {ASK_URL}")
emit(f"Model:    {MODEL}")

try:
    health = requests.get(HEALTH_URL, timeout=15).json()
    services = health.get("services", {})
    emit(f"API health: {health.get('status')}")
    for name, info in services.items():
        emit(f"  - {name}: {info.get('status')} ({info.get('message')})")
    if "bedrock" not in services:
        emit("  ! /health reports no 'bedrock' service - the API is running pre-Bedrock code.")
except Exception as e:
    emit(f"API health: unreachable ({e})")

# --- Part 1: single end-to-end question -----------------------------------
emit()
emit("PART 1: END-TO-END QUESTION")
emit("-" * 40)
emit(f"Question: {QUESTION}")

try:
    response_time, response = ask(QUESTION, top_k=5, use_hybrid=True)

    if response.status_code == 200:
        data = response.json()

        emit(f"\n✓ Success! ({response_time:.1f} seconds)")
        emit("\nAnswer:")
        emit("-" * 40)
        emit(data["answer"])
        emit("-" * 40)

        emit(f"\nSources: {len(data.get('sources', []))} papers")
        for source in data.get("sources", []):
            emit(f"  - {source}")
        emit(f"Chunks used: {data.get('chunks_used', 0)}")
        emit(f"Search mode: {data.get('search_mode', 'unknown')}")
    else:
        emit(f"\n✗ Request failed: HTTP {response.status_code}")
        emit(f"Response: {response.text[:200]}")

except Exception as e:
    emit(f"\n✗ Error: {e}")



'''
# --- Part 2: time profile -------------------------------------------------
emit()
emit("PART 2: TIME PROFILE")
emit("-" * 40)
emit(f"{TIMING_RUNS} runs per scenario; retrieval measured separately via /hybrid-search")
emit()

profile = []

for label, top_k, use_hybrid in SCENARIOS:
    ask_samples = []
    search_samples = []
    failures = 0

    for _ in range(TIMING_RUNS):
        try:
            elapsed, response = ask(QUESTION, top_k=top_k, use_hybrid=use_hybrid)
            if response.status_code == 200:
                ask_samples.append(elapsed)
            else:
                failures += 1
        except Exception:
            failures += 1

        try:
            elapsed, response = search_only(QUESTION, size=top_k, use_hybrid=use_hybrid)
            if response.status_code == 200:
                search_samples.append(elapsed)
        except Exception:
            pass

    if not ask_samples:
        emit(f"{label:<18} ✗ all {TIMING_RUNS} runs failed")
        profile.append({"label": label, "failed": True})
        continue

    ask_stats = summarize(ask_samples)
    search_stats = summarize(search_samples) if search_samples else None
    # Retrieval and generation are sequential inside /ask, so the difference is
    # a fair estimate of the Bedrock call (plus prompt assembly).
    generation = ask_stats["mean"] - search_stats["mean"] if search_stats else None

    emit(
        f"{label:<18} ask min={ask_stats['min']:6.2f}s  mean={ask_stats['mean']:6.2f}s  max={ask_stats['max']:6.2f}s"
        + (f"  | retrieval={search_stats['mean']:5.2f}s  generation≈{generation:6.2f}s" if search_stats else "")
        + (f"  | {failures} failed" if failures else "")
    )

    profile.append(
        {
            "label": label,
            "failed": False,
            "ask": ask_stats,
            "search": search_stats,
            "generation": generation,
            "failures": failures,
        }
    )

# --- Report ---------------------------------------------------------------
_os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now()
report_path = _os.path.join(RESULTS_DIR, f"rag_api_profile_{timestamp:%Y%m%d_%H%M%S}.md")

with open(report_path, "w") as f:
    f.write(f"# RAG /ask API profile\n\n")
    f.write(f"- Generated: {timestamp:%Y-%m-%d %H:%M:%S}\n")
    f.write(f"- Endpoint: `{ASK_URL}`\n")
    f.write(f"- Model: `{MODEL}`\n")
    f.write(f"- Runs per scenario: {TIMING_RUNS}\n\n")

    f.write("## Latency by scenario\n\n")
    f.write("| Scenario | min (s) | mean (s) | median (s) | max (s) | retrieval (s) | generation (s) |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for row in profile:
        if row["failed"]:
            f.write(f"| {row['label']} | - | - | - | - | - | all runs failed |\n")
            continue
        a = row["ask"]
        retrieval = f"{row['search']['mean']:.2f}" if row["search"] else "-"
        generation = f"{row['generation']:.2f}" if row["generation"] is not None else "-"
        f.write(
            f"| {row['label']} | {a['min']:.2f} | {a['mean']:.2f} | {a['median']:.2f} | "
            f"{a['max']:.2f} | {retrieval} | {generation} |\n"
        )

    f.write("\n## Run log\n\n```\n")
    f.write("\n".join(report_lines))
    f.write("\n```\n")

emit()
emit(f"Report saved: {report_path}")
'''