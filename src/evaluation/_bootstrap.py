"""Import-path bootstrap for every evaluation entry point.

Parts of ``src/services/indexing`` import their siblings without the ``src.``
prefix — ``hybrid_indexer.py`` does ``from services.opensearch.client import
OpenSearchClient``. That resolves only when ``src/`` is itself on ``sys.path``,
which is true inside the Airflow container (``PYTHONPATH=/opt/airflow/src`` in
``compose.yml``) but not for a script run from the repo root.

So both the repo root *and* ``src/`` have to be importable. Every module under
``src/evaluation`` imports this before touching anything in ``src.services``.
"""

import sys
from pathlib import Path

# .../production_rag/src/evaluation/_bootstrap.py -> .../production_rag
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"


def bootstrap() -> Path:
    """Put the repo root and ``src/`` on ``sys.path``. Idempotent.

    :returns: The repository root path.
    """
    for path in (SRC_ROOT, REPO_ROOT):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return REPO_ROOT


bootstrap()
