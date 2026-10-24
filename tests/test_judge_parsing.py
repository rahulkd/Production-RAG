"""Tests for judge reply parsing, validation, caching and calibration maths.

Hermetic: the judge's LLM is replaced by a stub that returns canned replies.
"""

import asyncio
from types import SimpleNamespace

import pytest
from src.evaluation.generation.calibrate import cohen_kappa
from src.evaluation.generation.judge import (
    Judge,
    JudgeError,
    JsonlCache,
    faithfulness_score,
    format_context,
    parse_judge_json,
    render,
    validate_claims,
    validate_refusal,
    validate_relevance,
    validate_verdicts,
)
from src.services.LLM.client import BedrockClient

# --- parse_judge_json -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"score": 2, "reason": "ok"}',
        '```json\n{"score": 2, "reason": "ok"}\n```',
        '```\n{"score": 2, "reason": "ok"}\n```',
        'Here is my verdict:\n{"score": 2, "reason": "ok"}\nThanks.',
    ],
)
def test_parse_accepts_fenced_unfenced_and_wrapped(text):
    assert parse_judge_json(text) == {"score": 2, "reason": "ok"}


@pytest.mark.parametrize("text", ["", "no json at all", '{"score": 2, "reason": ', '["a", "b"]'])
def test_parse_rejects_garbage(text):
    with pytest.raises(JudgeError):
        parse_judge_json(text)


# --- validators -------------------------------------------------------------


def test_relevance_accepts_int_and_digit_string():
    assert validate_relevance({"score": 3, "reason": "r"})["score"] == 3
    assert validate_relevance({"score": "1"})["score"] == 1


@pytest.mark.parametrize("score", [4, -1, 2.5, "high", None, True])
def test_relevance_rejects_out_of_scale(score):
    with pytest.raises(JudgeError):
        validate_relevance({"score": score})


def test_refusal_normalizes_case_and_rejects_unknown():
    assert validate_refusal({"label": " Refused "})["label"] == "refused"
    with pytest.raises(JudgeError):
        validate_refusal({"label": "partial"})


def test_claims_drops_blanks_and_requires_list():
    assert validate_claims({"claims": ["a", " ", "b"]}) == ["a", "b"]
    assert validate_claims({"claims": []}) == []
    with pytest.raises(JudgeError):
        validate_claims({"claims": "a"})


def test_verdicts_reordered_by_claim_id():
    payload = {
        "verdicts": [
            {"claim_id": 2, "label": "contradicted", "reason": "b"},
            {"claim_id": 1, "label": "supported", "reason": "a"},
        ]
    }
    assert [v["label"] for v in validate_verdicts(payload, 2)] == ["supported", "contradicted"]


def test_verdicts_count_mismatch_is_an_error():
    with pytest.raises(JudgeError):
        validate_verdicts({"verdicts": [{"claim_id": 1, "label": "supported"}]}, 2)


def test_verdicts_unknown_label_is_an_error():
    with pytest.raises(JudgeError):
        validate_verdicts({"verdicts": [{"claim_id": 1, "label": "partially"}]}, 1)


def test_faithfulness_counts_contradictions_separately():
    scores = faithfulness_score([{"label": "supported"}, {"label": "contradicted"}, {"label": "not_in_context"}, {"label": "supported"}])
    assert scores == {"score": 0.5, "claims": 4, "supported": 2, "contradicted": 1, "not_in_context": 1}
    assert faithfulness_score([])["score"] is None


# --- rendering --------------------------------------------------------------


def test_render_leaves_json_braces_alone():
    template = 'Q: {question}\n{"score": 0}\nA: {answer}'
    assert render(template, question="why {x}?", answer="because") == 'Q: why {x}?\n{"score": 0}\nA: because'


def test_format_context_matches_rag_prompt_labels():
    chunks = [{"arxiv_id": "2401.00001v1", "chunk_text": "alpha"}, {"arxiv_id": "2402.00002v1", "chunk_text": "beta"}]
    assert format_context(chunks) == "[1. arXiv:2401.00001v1]\nalpha\n\n[2. arXiv:2402.00002v1]\nbeta"


# --- Judge with a stub LLM --------------------------------------------------


class StubLLM:
    """Returns canned replies in order and records every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def generate(self, model, prompt, **kwargs):
        self.calls.append({"model": model, "prompt": prompt, **kwargs})
        return {"response": self.replies.pop(0), "stop_reason": "end_turn", "usage": {"inputTokens": 10, "outputTokens": 5}}


def make_judge(tmp_path, replies, read_cache=True):
    llm = StubLLM(replies)
    judge = Judge(
        llm,
        model="judge-model",
        params={"temperature": 0.0, "top_p": None},
        max_tokens=100,
        prompt_version="v1",
        cache=JsonlCache(tmp_path / "cache.jsonl"),
        read_cache=read_cache,
    )
    return judge, llm


def test_judge_caches_by_prompt_and_survives_reload(tmp_path):
    judge, llm = make_judge(tmp_path, ['{"label": "attempted", "reason": "r"}'])
    first = asyncio.run(judge.refusal("q?", "an answer"))
    second = asyncio.run(judge.refusal("q?", "an answer"))
    assert first == second == {"label": "attempted", "reason": "r"}
    assert len(llm.calls) == 1 and judge.cache_hits == 1

    reloaded, reloaded_llm = make_judge(tmp_path, [])
    assert asyncio.run(reloaded.refusal("q?", "an answer"))["label"] == "attempted"
    assert reloaded_llm.calls == []


def test_judge_passes_sampling_params_through(tmp_path):
    judge, llm = make_judge(tmp_path, ['{"score": 3, "reason": "r"}'])
    asyncio.run(judge.relevance("q?", "ref", "ans"))
    assert llm.calls[0]["temperature"] == 0.0
    assert llm.calls[0]["top_p"] is None
    assert llm.calls[0]["max_tokens"] == 100


def test_judge_cache_bypass_still_calls_model(tmp_path):
    judge, _ = make_judge(tmp_path, ['{"label": "refused", "reason": "r"}'])
    asyncio.run(judge.refusal("q?", "a"))
    bypass, llm = make_judge(tmp_path, ['{"label": "refused", "reason": "r"}'], read_cache=False)
    asyncio.run(bypass.refusal("q?", "a"))
    assert len(llm.calls) == 1


def test_unparseable_reply_is_not_cached(tmp_path):
    judge, _ = make_judge(tmp_path, ["not json"])
    with pytest.raises(JudgeError):
        asyncio.run(judge.refusal("q?", "a"))
    assert JsonlCache(tmp_path / "cache.jsonl").entries == {}


def test_faithfulness_prompt_never_contains_reference_answer(tmp_path):
    judge, llm = make_judge(tmp_path, ['{"verdicts": [{"claim_id": 1, "label": "supported", "reason": "r"}]}'])
    verdicts = asyncio.run(judge.verify("[1. arXiv:2401.00001v1]\nevidence", ["claim one"]))
    assert verdicts == [{"claim": "claim one", "label": "supported", "reason": "r"}]
    prompt = llm.calls[0]["prompt"]
    assert "evidence" in prompt and "1. claim one" in prompt
    assert "Reference answer" not in prompt


# --- BedrockClient sampling params ------------------------------------------


def test_inference_config_omits_none_sampling_params():
    defaults = SimpleNamespace(max_tokens=1024, temperature=0.5, top_p=0.9)
    assert BedrockClient._build_inference_config(defaults) == {"maxTokens": 1024, "temperature": 0.5, "topP": 0.9}
    assert BedrockClient._build_inference_config(defaults, temperature=0.0, top_p=None) == {"maxTokens": 1024, "temperature": 0.0}
    assert BedrockClient._build_inference_config(defaults, temperature=None, top_p=None, max_tokens=50) == {"maxTokens": 50}


# --- Cohen's kappa ----------------------------------------------------------


def test_kappa_perfect_agreement():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_kappa_single_shared_label_is_perfect():
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0


def test_kappa_hand_worked():
    # 10 items: both "y" on 4, both "n" on 3, disagree on 3.
    human = ["y"] * 4 + ["n"] * 3 + ["y", "y", "n"]
    judge = ["y"] * 4 + ["n"] * 3 + ["n", "n", "y"]
    # p_o = 0.7; human y=6 n=4, judge y=5 n=5; p_e = (30 + 20) / 100 = 0.5; kappa = 0.2 / 0.5
    assert cohen_kappa(human, judge) == pytest.approx(0.4)


def test_kappa_rejects_misaligned_input():
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])


def test_parse_repairs_latex_backslashes_in_reasons():
    # Verbatim shape of a DeepSeek reply quoting a paper (q0008, Phase 5 smoke run).
    text = '{"verdicts": [{"claim_id": 1, "label": "supported", "reason": "Abstract states \\\'up to $0.85\\%$ on GLUE\\\'."}]}'
    parsed = parse_judge_json(text)
    assert parsed["verdicts"][0]["label"] == "supported"
    assert "0.85" in parsed["verdicts"][0]["reason"]


def test_parse_keeps_valid_escapes_untouched():
    assert parse_judge_json('{"reason": "line\\nbreak \\"quoted\\" \\u00e9"}')["reason"] == 'line\nbreak "quoted" é'


def test_repair_does_not_split_an_escaped_backslash():
    # A literal backslash (valid "\\\\") next to an invalid "\\%" in the same reply.
    text = '{"reason": "path C:\\\\data and 5\\% gain"}'
    assert parse_judge_json(text)["reason"] == "path C:\\data and 5\\% gain"
