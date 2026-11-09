"""Online LLM-as-judge: scores live answers on their Langfuse traces.

Reuses the Phase 4 judge (``src/evaluation/generation/judge.py``) and its
refusal / claims / faithfulness rubrics unchanged, so live scores mean the same
as the offline evaluation's. Relevance uses a reference-free rubric
(``prompts/answer_relevance_v1.txt``): live traffic has no golden answer.

Judging runs as a background task after the response is sent. Its verdicts go
on the request's ``rag_request`` trace as scores:

* ``judge_refused``          BOOLEAN — the answer declined to answer
* ``judge_answer_relevance`` NUMERIC 0-3 — attempted answers only
* ``judge_faithfulness``     NUMERIC 0-1 — supported claims / all claims, attempted answers only

The judge calls themselves (prompts, replies, tokens, cost) are recorded in a
separate ``rag_judge`` trace with id ``<rag_request trace id>-judge``. Keeping
them out of ``rag_request`` leaves that trace's latency at what the user waited,
and lets the dashboard split generator cost from judge cost by trace name.
"""

import asyncio
import contextvars
import importlib.util
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import Settings
from src.services.langfuse.client import LangfuseTracer
from src.services.langfuse.tracer import bedrock_usage

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


# Per-judgment state for _TracedLLM. Each judgment runs in its own asyncio task,
# which gets its own copy of the context, so concurrent judgments never mix.
_judge_trace: contextvars.ContextVar = contextvars.ContextVar("judge_trace", default=None)
_judge_step: contextvars.ContextVar = contextvars.ContextVar("judge_step", default="judge")


class _TracedLLM:
    """Wraps the Bedrock client so each judge call becomes a generation in the current rag_judge trace.

    The offline ``Judge`` calls ``llm.generate(...)`` and is reused unchanged;
    only the object it calls is swapped.
    """

    def __init__(self, llm, tracer: LangfuseTracer, input_price_per_1m: float, output_price_per_1m: float):
        self.llm = llm
        self.tracer = tracer
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m

    async def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        generation = self.tracer.create_generation(
            trace=_judge_trace.get(),
            name=f"judge_{_judge_step.get()}",
            model=model,
            input_data={"prompt": prompt},
        )
        try:
            result = await self.llm.generate(model=model, prompt=prompt, **kwargs)
            self.tracer.update_generation(
                generation,
                output=result.get("response"),
                usage=bedrock_usage(result.get("usage"), self.input_price_per_1m, self.output_price_per_1m),
            )
            return result
        except Exception as e:
            self.tracer.update_span(generation, level="ERROR", status_message=str(e))
            raise
        finally:
            if generation:
                generation.end()


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
            _TracedLLM(
                llm,
                tracer,
                self.settings.judge_input_price_per_1m_tokens,
                self.settings.judge_output_price_per_1m_tokens,
            ),
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
        judge_trace = self.tracer.create_trace(
            name="rag_judge",
            trace_id=f"{trace_id}-judge",
            input_data={"question": question, "answer": answer},
            metadata={"parent_trace_id": trace_id, "judge_model": self.settings.judge_model, "rubric_version": RUBRIC_VERSION},
            tags=["judge"],
        )
        _judge_trace.set(judge_trace)
        verdict: Dict[str, Any] = {}

        try:
            _judge_step.set("refusal")
            refusal = await self.judge.refusal(question, answer)
            refused = refusal["label"] == "refused"
            verdict["refused"] = refused
            self.tracer.score_by_id(trace_id, "judge_refused", 1 if refused else 0, comment=refusal.get("reason"), data_type="BOOLEAN")
            if refused:
                # Relevance and faithfulness are meaningless for a refusal
                return

            _judge_step.set("answer_relevance")
            relevance = validate_relevance(await self.judge.ask("answer_relevance", question=question, answer=answer))
            verdict["answer_relevance"] = relevance["score"]
            self.tracer.score_by_id(
                trace_id, "judge_answer_relevance", relevance["score"], comment=relevance.get("reason"), data_type="NUMERIC"
            )

            _judge_step.set("claims")
            claims = await self.judge.claims(question, answer)
            if not claims:
                return
            _judge_step.set("faithfulness")
            verdicts = await self.judge.verify(format_context(chunks), claims)
            faithfulness = faithfulness_score(verdicts)
            verdict["faithfulness"] = faithfulness
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
            verdict["error"] = str(e)
            logger.warning(f"Online judge failed for trace {trace_id}: {e}")
        finally:
            if judge_trace:
                try:
                    judge_trace.update(output=verdict)
                except Exception as e:
                    logger.error(f"Error updating judge trace: {e}")
