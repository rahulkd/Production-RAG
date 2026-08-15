"""Phase 1: build and freeze the evaluation corpus.

Two steps, run in order:

1. ``python -m src.evaluation.corpus.discover`` — query arXiv per topic, select
   a deduplicated and balanced set of papers, write ``corpus_manifest.json``.
2. ``python -m src.evaluation.corpus.build`` — fetch those exact IDs, parse the
   PDFs, store to Postgres, chunk + embed, and index into the evaluation index.

Discovery is separated from building so the corpus is defined by a committed
list of arXiv IDs rather than by a query that returns something different
every day.
"""
