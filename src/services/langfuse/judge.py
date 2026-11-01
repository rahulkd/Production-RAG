"""Online LLM-as-judge: scores live answers on their Langfuse traces.

Reuses the Phase 4 judge (``src/evaluation/generation/judge.py``) and its
refusal / claims / faithfulness rubrics unchanged, so live scores mean the same
as the offline evaluation's. Relevance uses a reference-free rubric
(``prompts/answer_relevance_v1.txt``): live traffic has no golden answer.

Judging runs as a background task after the response is sent. It never adds
spans to the trace (that would stretch the trace's latency), only scores:

* ``judge_refused``          BOOLEAN — the answer declined to answer
* ``judge_answer_relevance`` NUMERIC 0-3 — attempted answers only
* ``judge_faithfulness``     NUMERIC 0-1 — supported claims / all claims, attempted answers only
"""

import asyncio
import importlib.util
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import Settings
from src.services.langfuse.client import LangfuseTracer

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
RUBRIC_VERSION = "v1"


def _load_offline_judge():
    """Load ``src/evaluation/generation/judge.py`` by path.

    A normal import would run ``src/evaluation/__init__.py``, whose bootstrap
    puts ``src/`` at the front of ``sys.path`` -- fine for the evaluation
    scripts, but inside the API it would let top-level names like ``config`` or
    ``models`` shadow installed packages. judge.py itself imports only stdlib.
    """
    path = Path(__file__).resolve().parents[2] / "evaluation" / "generation" / "judge.py"
    spec = importlib.util.spec_from_file_location("rag_offline_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_offline = _load_offline_judge()
Judge = _offline.Judge
JsonlCache = _offline.JsonlCache
faithfulness_score = _offline.faithfulness_score
format_context = _offline.format_context
validate_relevance = _offline.validate_relevance


class _NoCache(JsonlCache):
    """The offline judge caches verdicts on disk; live answers are judged once, so keep nothing."""

    def __init__(self):
        self.entries: Dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        pass


class OnlineJudge:
    """Schedules background judging of answers and writes the verdicts as Langfuse scores.

    :param llm: ``BedrockClient`` (the judge model is called through it).
    :param tracer: ``LangfuseTracer`` the scores are sent through.
    :param settings: App settings (``settings.langfuse.judge_*``).
    """

    def __init__(self, llm, tracer: LangfuseTracer, settings: Settings):
        self.tracer = tracer
        self.settings = settings.langfuse
        self.enabled = self.settings.judge_enabled and tracer.client is not None
        self.judge = Judge(
            llm,
            model=self.settings.judge_model,
            # Same sampling as the offline judge (JUDGE_PARAMS): temperature 0, top_p omitted
            params={"temperature": 0.0, "top_p": None},
            max_tokens=self.settings.judge_max_tokens,
            prompt_version=RUBRIC_VERSION,
            cache=_NoCache(),
        )
        self.judge.templates["answer_relevance"] = (PROMPTS_DIR / f"answer_relevance_{RUBRIC_VERSION}.txt").read_text(encoding="utf-8")
        # Strong references, so pending tasks are not garbage-collected mid-run
        self._tasks: Set[asyncio.Task] = set()

        if self.enabled:
            logger.info(f"Online judge enabled (model: {self.settings.judge_model}, sample rate: {self.settings.judge_sample_rate})")
        else:
            logger.info("Online judge disabled")

    def schedule(self, trace_id: Optional[str], question: str, answer: str, chunks: List[Dict[str, Any]]) -> None:
        """Judge an answer in the background, if enabled and sampled. Returns immediately."""
        if not (self.enabled and trace_id and answer and chunks):
            return
        if random.random() >= self.settings.judge_sample_rate:
            return

        task = asyncio.create_task(self._judge(trace_id, question, answer, chunks))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self, timeout: float = 30.0) -> None:
        """Wait for pending judgments on shutdown, so their scores are sent before the final flush."""
        if not self._tasks:
            return
        logger.info(f"Waiting for {len(self._tasks)} pending judgment(s)")
        _, pending = await asyncio.wait(set(self._tasks), timeout=timeout)
        for task in pending:
            task.cancel()

    async def _judge(self, trace_id: str, question: str, answer: str, chunks: List[Dict[str, Any]]) -> None:
        try:
            refusal = await self.judge.refusal(question, answer)
            refused = refusal["label"] == "refused"
            self.tracer.score_by_id(trace_id, "judge_refused", 1 if refused else 0, comment=refusal.get("reason"), data_type="BOOLEAN")
            if refused:
                # Relevance and faithfulness are meaningless for a refusal
                return

            relevance = validate_relevance(await self.judge.ask("answer_relevance", question=question, answer=answer))
            self.tracer.score_by_id(
                trace_id, "judge_answer_relevance", relevance["score"], comment=relevance.get("reason"), data_type="NUMERIC"
            )

            claims = await self.judge.claims(question, answer)
            if not claims:
                return
            verdicts = await self.judge.verify(format_context(chunks), claims)
            faithfulness = faithfulness_score(verdicts)
            self.tracer.score_by_id(
                trace_id,
                "judge_faithfulness",
                faithfulness["score"],
                comment=(
                    f"{faithfulness['supported']}/{faithfulness['claims']} claims supported, "
                    f"{faithfulness['contradicted']} contradicted, {faithfulness['not_in_context']} not in context"
                ),
                data_type="NUMERIC",
            )
        except Exception as e:
            logger.warning(f"Online judge failed for trace {trace_id}: {e}")
