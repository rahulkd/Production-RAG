"""Correctness checks for Jina embeddings.

Reuses the ``JinaEmbeddingsGenerator`` from ``test_embedding_api.py`` and verifies
that the embeddings it returns are actually usable — not just that the call
succeeds. Runs five independent checks:

  1. Structure      - right count, expected dimension (1024), plain floats.
  2. Not-dummy      - not the ``[0.1] * dim`` fallback (which means the API call
                      silently failed / no API key).
  3. Numeric health - all values finite (no NaN/Inf) and unit-normalized
                      (Jina v3 returns L2-normalized vectors).
  4. Semantic order - semantically related texts are more similar to each other
                      than to an unrelated text. This is the real "are these
                      meaningful?" test.
  5. Determinism    - embedding the same text twice yields the same vector.

Exit code is 0 only if every check passes, so this doubles as a CI smoke test.
Set ``JINA_API_KEY`` in the environment before running — dummy embeddings cannot
pass the semantic check.
"""

import asyncio
import math
import os
import sys

import numpy as np

# Import the generator from the sibling reference file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_embedding_api import JinaEmbeddingsGenerator  # noqa: E402

EXPECTED_DIM = 1024
DUMMY_VALUE = 0.1  # JinaEmbeddingsGenerator's fallback fills vectors with this.


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class CheckReporter:
    """Tiny PASS/FAIL harness so the script can report all checks + an exit code."""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        status = "PASS" if condition else "FAIL"
        line = f"  [{status}] {name}"
        if detail:
            line += f"  ->  {detail}"
        print(line)
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        return condition

    def summary(self) -> bool:
        total = self.passed + self.failed
        print("=" * 60)
        print(f"RESULT: {self.passed}/{total} checks passed")
        return self.failed == 0


def is_dummy(vec) -> bool:
    """True if a vector looks like the [0.1]*dim fallback."""
    arr = np.asarray(vec, dtype=np.float64)
    return bool(np.allclose(arr, DUMMY_VALUE))


async def run_checks() -> bool:
    print("VERIFYING JINA EMBEDDING CORRECTNESS")
    print("=" * 60)

    reporter = CheckReporter()

    generator = JinaEmbeddingsGenerator()
    if not generator.api_key:
        print(
            "\n[!] No JINA_API_KEY set - the generator will return DUMMY embeddings.\n"
            "    Structure checks may pass but semantic checks will fail.\n"
            "    Set JINA_API_KEY to run a real verification.\n"
        )

    # Two related (ML/AI) texts and one unrelated (cooking) text.
    related_a = "Machine learning is a subset of artificial intelligence."
    related_b = "Neural networks are computational models used in deep learning."
    unrelated = "The recipe calls for two cups of flour and a pinch of salt."
    texts = [related_a, related_b, unrelated]

    embeddings = await generator.generate_embeddings(texts)

    # --- 1. Structure ------------------------------------------------------
    print("\n1. Structure")
    reporter.check(
        "returned one embedding per input",
        isinstance(embeddings, list) and len(embeddings) == len(texts),
        f"expected {len(texts)}, got {len(embeddings) if embeddings else 0}",
    )
    dims = {len(e) for e in embeddings} if embeddings else set()
    reporter.check(
        f"every embedding has dimension {EXPECTED_DIM}",
        dims == {EXPECTED_DIM},
        f"dimensions seen: {sorted(dims)}",
    )
    all_floats = all(isinstance(v, (int, float)) for e in embeddings for v in e)
    reporter.check("all values are numeric", all_floats)

    # --- 2. Not-dummy ------------------------------------------------------
    print("\n2. Not the dummy fallback")
    any_dummy = any(is_dummy(e) for e in embeddings)
    reporter.check(
        "embeddings are not the [0.1]*dim fallback",
        not any_dummy,
        "got dummy vectors (API call failed or no key)" if any_dummy else "real vectors",
    )

    # --- 3. Numeric health -------------------------------------------------
    print("\n3. Numeric health")
    arr = np.asarray(embeddings, dtype=np.float64) if embeddings else np.empty((0,))
    finite = bool(np.isfinite(arr).all()) if arr.size else False
    reporter.check("no NaN or Inf values", finite)
    norms = np.linalg.norm(arr, axis=1) if arr.size else np.array([])
    normalized = bool(np.allclose(norms, 1.0, atol=1e-2)) if norms.size else False
    reporter.check(
        "vectors are L2-normalized (Jina v3 default)",
        normalized,
        f"norms ~ {np.round(norms, 4).tolist()}" if norms.size else "n/a",
    )

    # --- 4. Semantic ordering ---------------------------------------------
    print("\n4. Semantic ordering")
    if len(embeddings) == 3 and not any_dummy:
        sim_related = cosine_similarity(embeddings[0], embeddings[1])
        sim_unrelated_a = cosine_similarity(embeddings[0], embeddings[2])
        sim_unrelated_b = cosine_similarity(embeddings[1], embeddings[2])
        reporter.check(
            "related pair more similar than unrelated pairs",
            sim_related > sim_unrelated_a and sim_related > sim_unrelated_b,
            f"related={sim_related:.4f}  unrelated=({sim_unrelated_a:.4f}, {sim_unrelated_b:.4f})",
        )
        reporter.check(
            "cosine similarities are within a valid [-1, 1] range",
            all(-1.0001 <= s <= 1.0001 for s in (sim_related, sim_unrelated_a, sim_unrelated_b)),
        )
    else:
        reporter.check(
            "related pair more similar than unrelated pairs",
            False,
            "skipped - need 3 real (non-dummy) embeddings",
        )

    # --- 5. Determinism ----------------------------------------------------
    print("\n5. Determinism")
    if not any_dummy:
        repeat = await generator.generate_embeddings([related_a])
        same = len(repeat) == 1 and math.isclose(
            cosine_similarity(embeddings[0], repeat[0]), 1.0, abs_tol=1e-4
        )
        reporter.check(
            "same text embeds to the same vector on repeat",
            same,
            f"self-similarity={cosine_similarity(embeddings[0], repeat[0]):.6f}" if repeat else "no result",
        )
    else:
        reporter.check("same text embeds to the same vector on repeat", False, "skipped - dummy vectors")

    print()
    return reporter.summary()


if __name__ == "__main__":
    ok = asyncio.run(run_checks())
    sys.exit(0 if ok else 1)
