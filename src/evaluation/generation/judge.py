"""LLM-as-judge for the generation analysis.

Rubrics are versioned text files under ``prompts/``. Every judgment is cached
by ``sha256(model + prompt version + rendered prompt)``, so a re-run after a
retrieval-only change pays only for prompts whose context actually changed.

Rules the plan requires, and where they are enforced:

* temperature 0 — the caller passes ``JUDGE_PARAMS`` through to ``generate``.
* JSON via the prompt (Converse has no schema parameter) — :func:`parse_judge_json`
  strips fences and falls back to the outermost object, like
  ``ResponseParser._extract_json_fallback``.
* One answer at a time, generator never named — no prompt receives more than one
  answer or the SUT model id.
* The faithfulness judge never sees the reference answer — :meth:`Judge.verify`
  takes only the retrieved context and the claims.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

FAITHFULNESS_LABELS = ("supported", "contradicted", "not_in_context")
REFUSAL_LABELS = ("refused", "attempted")
RELEVANCE_SCORES = (0, 1, 2, 3)

# Every backslash escape, consumed as a pair left to right so an escaped
# backslash ("\\\\") is never split.
_ESCAPE_PAIR = re.compile(r"\\(.)", re.DOTALL)
_JSON_ESCAPES = set('"\\/bfnrtu')


def _repair_escapes(text: str) -> str:
    """Double every backslash that does not start a valid JSON escape.

    :param text: Raw JSON-ish text.
    :returns: Text where ``\\%`` becomes ``\\\\%`` and valid escapes are untouched.
    """
    return _ESCAPE_PAIR.sub(lambda m: m.group(0) if m.group(1) in _JSON_ESCAPES else "\\\\" + m.group(1), text)


class JudgeError(ValueError):
    """The judge's reply could not be parsed or failed validation."""


def load_prompt(name: str, version: str) -> str:
    """Read a rubric template.

    :param name: Template stem, e.g. ``"faithfulness"``.
    :param version: Prompt version, e.g. ``"v1"``.
    :returns: Template text.
    :raises FileNotFoundError: If the template is missing.
    """
    path = PROMPTS_DIR / f"{name}_{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Judge prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def render(template: str, **fields: str) -> str:
    """Fill ``{name}`` placeholders.

    ``str.format`` would choke on the braces in the JSON examples and in the
    excerpts, so placeholders are replaced literally.

    :param template: Template text.
    :param fields: Placeholder values.
    :returns: Rendered prompt.
    """
    for name, value in fields.items():
        template = template.replace("{" + name + "}", value)
    return template


def format_context(chunks: Sequence[Dict[str, Any]]) -> str:
    """Render retrieved chunks exactly as ``RAGPromptBuilder.create_rag_prompt`` does.

    The judge must see the same evidence the generator saw, labelled the same way.

    :param chunks: ``{arxiv_id, chunk_text}`` dicts, in prompt order.
    :returns: Context block.
    """
    return "".join(f"[{i}. arXiv:{chunk['arxiv_id']}]\n{chunk['chunk_text']}\n\n" for i, chunk in enumerate(chunks, 1)).rstrip()


def parse_judge_json(text: str) -> Dict[str, Any]:
    """Parse a judge reply into a JSON object.

    :param text: Raw model output.
    :returns: The parsed object.
    :raises JudgeError: If no JSON object can be recovered.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    # Reasons quote the papers, LaTeX included ("$0.85\%$", "\'"), and a backslash
    # before anything but a JSON escape character is invalid JSON. Retry with
    # those backslashes doubled rather than lose the verdict.
    candidates += [_repair_escapes(candidate) for candidate in list(candidates)]

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise JudgeError(f"No JSON object in judge output: {text[:200]!r}")


def validate_refusal(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Check a refusal verdict.

    :param payload: Parsed reply.
    :returns: ``{"label", "reason"}``.
    :raises JudgeError: On an unknown label.
    """
    label = str(payload.get("label", "")).strip().lower()
    if label not in REFUSAL_LABELS:
        raise JudgeError(f"refusal label {payload.get('label')!r} not in {REFUSAL_LABELS}")
    return {"label": label, "reason": str(payload.get("reason", ""))}


def validate_claims(payload: Dict[str, Any]) -> List[str]:
    """Check a claim decomposition.

    :param payload: Parsed reply.
    :returns: Non-empty claim strings.
    :raises JudgeError: If ``claims`` is not a list.
    """
    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise JudgeError(f"claims must be a list, got {type(claims).__name__}")
    return [str(claim).strip() for claim in claims if str(claim).strip()]


def validate_verdicts(payload: Dict[str, Any], claim_count: int) -> List[Dict[str, str]]:
    """Check claim verdicts, one per claim, in order.

    :param payload: Parsed reply.
    :param claim_count: Number of claims that were sent.
    :returns: ``{"label", "reason"}`` per claim.
    :raises JudgeError: On a count mismatch or an unknown label.
    """
    verdicts = payload.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != claim_count:
        got = len(verdicts) if isinstance(verdicts, list) else type(verdicts).__name__
        raise JudgeError(f"expected {claim_count} verdicts, got {got}")

    # Order by claim_id when the judge supplied it, so a reordered reply still
    # lines up with the claims.
    if all(isinstance(verdict, dict) and isinstance(verdict.get("claim_id"), int) for verdict in verdicts):
        verdicts = sorted(verdicts, key=lambda verdict: verdict["claim_id"])

    checked = []
    for verdict in verdicts:
        label = str(verdict.get("label", "")).strip().lower() if isinstance(verdict, dict) else ""
        if label not in FAITHFULNESS_LABELS:
            raise JudgeError(f"faithfulness label {label!r} not in {FAITHFULNESS_LABELS}")
        checked.append({"label": label, "reason": str(verdict.get("reason", ""))})
    return checked


def validate_relevance(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Check a relevance score.

    :param payload: Parsed reply.
    :returns: ``{"score", "reason"}``.
    :raises JudgeError: If the score is not 0-3.
    """
    score = payload.get("score")
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score.strip())
    if isinstance(score, bool) or score not in RELEVANCE_SCORES:
        raise JudgeError(f"relevance score {payload.get('score')!r} not in {RELEVANCE_SCORES}")
    return {"score": int(score), "reason": str(payload.get("reason", ""))}


class JsonlCache:
    """Append-only key/value cache on disk.

    Appending one line per entry means a crashed run keeps every judgment it
    already paid for.

    :param path: JSONL file of ``{"key", "value"}`` rows.
    """

    def __init__(self, path: Path):
        self.path = path
        self.entries: Dict[str, Any] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.entries[row["key"]] = row["value"]

    @staticmethod
    def key(*parts: str) -> str:
        """Cache key for a request.

        :param parts: Model, version, prompt — anything that changes the output.
        :returns: sha256 hex digest.
        """
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Cached value, if any.

        :param key: From :meth:`key`.
        :returns: The value, or None.
        """
        return self.entries.get(key)

    def put(self, key: str, value: Any) -> None:
        """Store a value and append it to disk.

        :param key: From :meth:`key`.
        :param value: JSON-serialisable value.
        """
        self.entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


class Judge:
    """Rubric-driven judge over a ``BedrockClient``.

    :param llm: ``BedrockClient``.
    :param model: Judge model id.
    :param params: Sampling kwargs for ``generate`` (``None`` omits a field).
    :param max_tokens: Reply budget.
    :param prompt_version: Rubric version; selects the files and keys the cache.
    :param cache: Verdict cache.
    :param read_cache: False to ignore cached verdicts (still writes new ones),
        for the run-twice determinism check.
    """

    def __init__(
        self,
        llm,
        model: str,
        params: Dict[str, Any],
        max_tokens: int,
        prompt_version: str,
        cache: JsonlCache,
        read_cache: bool = True,
    ):
        self.llm = llm
        self.model = model
        self.params = params
        self.max_tokens = max_tokens
        self.prompt_version = prompt_version
        self.cache = cache
        self.read_cache = read_cache
        self.templates = {name: load_prompt(name, prompt_version) for name in ("refusal", "claims", "faithfulness", "relevance")}
        self.calls = 0
        self.cache_hits = 0
        self.usage = {"inputTokens": 0, "outputTokens": 0}

    async def ask(self, rubric: str, **fields: str) -> Dict[str, Any]:
        """Render a rubric, call the judge (or the cache), and parse the reply.

        Only parseable replies are cached, so a malformed reply is retried on
        the next run rather than frozen in.

        :param rubric: Template name.
        :param fields: Placeholder values.
        :returns: Parsed JSON object.
        :raises JudgeError: If the reply has no JSON object.
        """
        prompt = render(self.templates[rubric], **fields)
        key = JsonlCache.key(self.model, self.prompt_version, prompt)

        cached = self.cache.get(key) if self.read_cache else None
        if cached is not None:
            self.cache_hits += 1
            return cached

        result = await self.llm.generate(model=self.model, prompt=prompt, max_tokens=self.max_tokens, **self.params)
        self.calls += 1
        for field_name in self.usage:
            self.usage[field_name] += int(result.get("usage", {}).get(field_name, 0))

        if result.get("stop_reason") == "max_tokens":
            raise JudgeError(f"{rubric} reply hit max_tokens={self.max_tokens}")
        parsed = parse_judge_json(result["response"])
        self.cache.put(key, parsed)
        return parsed

    async def refusal(self, question: str, answer: str) -> Dict[str, Any]:
        """Classify an answer as refused or attempted (Q9).

        :param question: The question.
        :param answer: The generated answer.
        :returns: ``{"label", "reason"}``.
        """
        return validate_refusal(await self.ask("refusal", question=question, answer=answer))

    async def claims(self, question: str, answer: str) -> List[str]:
        """Decompose an answer into atomic claims (faithfulness call 1).

        :param question: The question, for resolving references.
        :param answer: The generated answer.
        :returns: Claims.
        """
        return validate_claims(await self.ask("claims", question=question, answer=answer))

    async def verify(self, context: str, claims: Sequence[str]) -> List[Dict[str, str]]:
        """Label each claim against the retrieved context only (faithfulness call 2).

        :param context: Output of :func:`format_context`.
        :param claims: Claims from :meth:`claims`.
        :returns: ``{"claim", "label", "reason"}`` per claim.
        """
        numbered = "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims, 1))
        verdicts = validate_verdicts(await self.ask("faithfulness", context=context, claims=numbered), len(claims))
        return [{"claim": claim, **verdict} for claim, verdict in zip(claims, verdicts)]

    async def relevance(self, question: str, reference_answer: str, answer: str) -> Dict[str, Any]:
        """Score how well an answer addresses the question, 0-3.

        :param question: The question.
        :param reference_answer: Golden reference answer.
        :param answer: The generated answer.
        :returns: ``{"score", "reason"}``.
        """
        return validate_relevance(await self.ask("relevance", question=question, reference_answer=reference_answer, answer=answer))


def faithfulness_score(verdicts: Sequence[Dict[str, str]]) -> Dict[str, Optional[float]]:
    """Aggregate claim verdicts for one answer.

    Contradictions are counted separately: one contradicted claim is worse than
    several merely unsupported ones, and folding it into the ratio hides that.

    :param verdicts: Output of :meth:`Judge.verify`.
    :returns: ``score`` (supported / total; None with no claims),
        ``claims``, ``supported``, ``contradicted``, ``not_in_context``.
    """
    counts = {label: sum(1 for verdict in verdicts if verdict["label"] == label) for label in FAITHFULNESS_LABELS}
    total = len(verdicts)
    return {
        "score": counts["supported"] / total if total else None,
        "claims": total,
        **counts,
    }
