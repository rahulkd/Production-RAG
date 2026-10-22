"""Phase 2: the golden question set.

Three steps, run in order:

1. ``python -m src.evaluation.dataset.generate`` — sample seed chunks, drive the
   generator, filter leaky questions, write ``golden_v1.draft.jsonl``.
2. ``python -m src.evaluation.dataset.review`` — accept / edit / reject each
   question against the chunk it came from.
3. ``python -m src.evaluation.dataset.review --promote`` — write the accepted
   rows to ``golden_v1.jsonl``.

Generation is separated from promotion because an unreviewed LLM-generated set
measures the question generator rather than the retriever under test.
"""
