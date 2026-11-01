import logging

from fastapi import APIRouter, HTTPException
from src.dependencies import LangfuseDep
from src.schemas.api.ask import FeedbackRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, langfuse_tracer: LangfuseDep) -> dict:
    """
    Record a user's thumbs up/down on an answer as a ``user_feedback`` score on its Langfuse trace.

    The score id is derived from the trace id, so changing the vote overwrites
    the earlier one instead of adding a second score.
    """
    recorded = langfuse_tracer.score_by_id(
        trace_id=feedback.trace_id,
        name="user_feedback",
        value=feedback.value,
        comment=feedback.comment,
        data_type="BOOLEAN",
        score_id=f"{feedback.trace_id}-user_feedback",
    )
    if not recorded:
        raise HTTPException(status_code=503, detail="Tracing is disabled; feedback was not recorded")

    logger.info(f"Recorded user feedback {feedback.value} for trace {feedback.trace_id}")
    return {"status": "recorded"}
