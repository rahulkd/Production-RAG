"""Evaluation harness for the arXiv RAG pipeline (cs.AI + cs.LG).

Layers are kept separate so that a failure is attributable to one stage:

* ``corpus``     — build and freeze the evaluation corpus  (Phase 1)
* ``dataset``    — golden question set                     (Phase 2)
* ``retrieval``  — exact retrieval metrics                 (Phase 3)
* ``generation`` — LLM-as-judge answer metrics             (Phase 4)

Every entry point is ``python -m src.evaluation.<module>``.

See ``docs/evaluation_plan.md`` for the full analysis plan.
"""

from src.evaluation._bootstrap import REPO_ROOT, bootstrap

__all__ = ["REPO_ROOT", "bootstrap"]
