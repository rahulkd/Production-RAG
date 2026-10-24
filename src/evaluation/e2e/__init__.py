"""Phase 5: end-to-end analysis.

``python -m src.evaluation.e2e.runner`` sends the golden set through the live
``/api/v1/ask`` and ``/api/v1/stream`` routes and checks that the deployed API
behaves like the in-process harness (Q11): same retrieved papers, hybrid search
mode on every response, latency, and Phase 4 answer quality on what is served.
"""
