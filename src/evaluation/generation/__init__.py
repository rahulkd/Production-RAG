"""Phase 4: generation analysis.

``python -m src.evaluation.generation.runner`` answers every accepted golden
question through the pinned ``/ask`` path and scores faithfulness, relevance,
citations and refusals, appending one row to the generation leaderboard.

* :mod:`.judge`     — rubric prompts, verdict cache, JSON parsing (LLM-as-judge)
* :mod:`.citations` — ``[arXiv:id]`` extraction and citation precision (no LLM)
* :mod:`.runner`    — retrieves, generates, judges, writes results
* :mod:`.calibrate` — blind human labels and Cohen's kappa for the judge
"""
