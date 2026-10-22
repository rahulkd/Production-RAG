"""Phase 3: retrieval analysis.

``python -m src.evaluation.retrieval.runner`` runs Sweep A — every retrieval
mode (bm25, vector, hybrid) at every cutoff (3, 5, 10, 20) — over the answerable
rows of the golden set, and appends one leaderboard row per config.

* :mod:`.metrics` — Recall@k, nDCG@k, MRR@k, Hit@k, paper Recall@k (pure functions)
* :mod:`.configs` — the sweep grid
* :mod:`.runner`  — embeds, retrieves, scores, writes results
"""
