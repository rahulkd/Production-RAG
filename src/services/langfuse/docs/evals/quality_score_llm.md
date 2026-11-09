# Online LLM-as-judge: why four calls per answer

Design note for the judge in [judge.py](../../judge.py) (`OnlineJudge`), which writes the `judge_*` quality scores onto Langfuse traces. It records why one answer costs up to four judge calls, and what changing that would trade away.

## What runs per answer

After an uncached `/ask` or `/stream` response is sent, `OnlineJudge.schedule` judges it in a background task. The verdicts are written as scores on the request's `rag_request` trace. The calls themselves are recorded as generations, with token usage and cost, in a separate `rag_judge` trace with id `<rag_request trace id>-judge` (see [Where judge cost shows up](#where-judge-cost-shows-up)).

| # | Call (rubric) | Sees | Approx. size | Score written |
|---|---|---|---|---|
| 1 | `refusal` | question + answer | small | `judge_refused` (BOOLEAN) |
| 2 | `answer_relevance` | question + answer | small | `judge_answer_relevance` (NUMERIC 0-3) |
| 3 | `claims` | question + answer | small | none; feeds call 4 |
| 4 | `faithfulness` (verify) | the retrieved chunks + the claims | ~10k tokens at `top_k=10` | `judge_faithfulness` (NUMERIC 0-1, supported / all claims) |

Flow:

```
refusal ──► refused? ──yes──► stop (relevance / faithfulness are meaningless for a refusal)
              │ no
              ├──► answer_relevance
              └──► claims ──► (no claims? stop) ──► faithfulness ──► score
```

So a refusal costs 1 call; an attempted answer costs 4 (3 if the answer has no checkable claims).

## Why four

Only faithfulness is inherently multi-step. It is the Phase 4 method: split the answer into atomic claims, then label each claim `supported` / `contradicted` / `not_in_context` against the retrieved context. That is two calls. Refusal and relevance are separate rubrics, each with its own prompt.

The rubrics are the offline evaluation's ([src/evaluation/generation/prompts/](../../../../evaluation/generation/prompts/)), reused unchanged for `refusal`, `claims` and `faithfulness`. Relevance is the one new rubric: [answer_relevance_v1.txt](../../prompts/answer_relevance_v1.txt), reference-free because live traffic has no golden answer.

Those three were calibrated in Phase 4 (blind human labels, Cohen's kappa). Reusing them means a live `judge_faithfulness` means the same thing as the number in the evaluation report.

## Question considered: why not one call for all three metrics?

A single prompt could score refusal, relevance and faithfulness together. It is possible, but the saving is smaller than it looks and the costs are real.

### What a single call would save

- **Money: little.** The three small calls (refusal, relevance, claims) see only the question and answer, so they cost very little. The expensive call is #4, whose ~10k-token context dominates. A combined prompt must also carry that context, so it costs about the same as call #4 alone.
- **What is really saved:** the number of API calls and a few seconds of background latency. Neither affects the user, because judging runs after the response.

### What it would cost

1. **Comparability with offline results.** The Phase 4 rubrics were calibrated against human labels. A new combined prompt is an uncalibrated judge, so live scores would no longer mean the same as the evaluation report.
2. **Faithfulness quality.** Asking for one overall "faithfulness" score in a single pass tends to produce a lenient score. Claim-by-claim checking is more reliable. The Phase 4 notes already record DeepSeek leaning lenient on a hard claim (`src/evaluation/config.py`, comment above `JUDGE_MODEL`).
3. **Metric interference.** One prompt covering three rubrics lets the model's judgement on one leak into the others, e.g. a refusal judgement colouring relevance.

## Measured cost (2026-11-09, one attempted answer, `top_k=10`)

| Trace / call | Model | Input → output tokens | Cost (USD) |
|---|---|---|---|
| `rag_request` / `llm_generation` | Llama 3.3 70B ($0.72 / $0.72 per 1M) | 12,022 → 156 | 0.008768 |
| `rag_judge` / `judge_refusal` | DeepSeek V3.2 ($0.62 / $1.85 per 1M) | 372 → 23 | 0.000273 |
| `rag_judge` / `judge_answer_relevance` | DeepSeek V3.2 | 405 → 20 | 0.000288 |
| `rag_judge` / `judge_claims` | DeepSeek V3.2 | 399 → 56 | 0.000351 |
| `rag_judge` / `judge_faithfulness` | DeepSeek V3.2 | 11,613 → 73 | 0.007335 |
| **Judge total** | | 12,961 | **0.008247** |

- Judging adds about **94 %** to the cost of an answer, almost doubling it.
- The faithfulness call is **89 %** of judge cost, because it carries the full retrieved context.
- The three small calls together cost about $0.0009.

This confirms the single-call analysis below with real numbers: merging calls saves under $0.001 per answer. The real lever is `LANGFUSE__JUDGE_SAMPLE_RATE`, or, to make faithfulness itself cheaper, a smaller context (lower `top_k`).

## Options and decision

| Option | Calls per attempted answer | Verdict |
|---|---|---|
| **A. Keep four calls** (current) | 4 | **Current choice.** Scores stay comparable to Phase 4. Control cost with `LANGFUSE__JUDGE_SAMPLE_RATE` (e.g. `0.2`) instead of changing the method |
| **B. Merge `refusal` + `answer_relevance`** | 3 | Safe merge: both need only question + answer, no context. Would need a new combined rubric, checked against labelled examples before trusting the scores |
| **C. One call for everything** | 1 | Fewest calls, but gives up all three costs above. If ever wanted, add it as a separate metric family (e.g. `quick_*` scores) rather than replacing the calibrated `judge_*` ones |

**Decision: stay on A, with a sample rate if cost matters.** Revisit B if the number of calls, rather than tokens, becomes the problem (rate limits, background task backlog).

## Where judge cost shows up

Judge cost is **not** in the `rag_request` trace. That trace keeps only the generator's cost and the latency the user waited for. It lives in a separate trace:

- **Name** `rag_judge`, **tag** `judge`, **id** `<rag_request trace id>-judge`.
- **Metadata:** `parent_trace_id`, `judge_model`, `rubric_version`.
- **Contents:** one generation per call (`judge_refusal`, `judge_answer_relevance`, `judge_claims`, `judge_faithfulness`), each with its prompt, reply, tokens and cost.
- **Output:** the verdicts.

To go from an answer to its judgment, append `-judge` to the trace id in the URL. In the dashboard, filter or group cost by trace name to split generator spend from judge spend. Prices come from `LANGFUSE__JUDGE_INPUT_PRICE_PER_1M_TOKENS` / `..._OUTPUT_...` (defaults $0.62 / $1.85, DeepSeek V3.2 on Bedrock US).

## Cost levers (no method change)

All in `.env`, read by `LangfuseSettings` in [config.py](../../../../config.py):

| Setting | Default | Effect |
|---|---|---|
| `LANGFUSE__JUDGE_SAMPLE_RATE` | `1.0` | Fraction of uncached answers judged. Lowering it cuts all four calls proportionally |
| `LANGFUSE__JUDGE_ENABLED` | `true` | Master switch. **Set to `false` during evaluation runs**: the e2e runner calls the live `/ask` and `/stream`, so leaving it on would pay for judging twice. Recreate the api container after changing it (`docker compose up -d --force-recreate api`). Also flush Redis (`docker exec rag-redis redis-cli FLUSHALL`) so runs get fresh answers |
| `LANGFUSE__JUDGE_MODEL` | `deepseek.v3.2` | Same judge model as Phase 4. Changing it breaks comparability with the calibrated rubrics |

Cached answers are never judged (a cache hit has no new answer), and a refusal stops after one call.

## If this is revisited

- Before changing any rubric, bump the prompt version (`RUBRIC_VERSION` in `judge.py` and `JUDGE_PROMPT_VERSION` offline) and re-run the Phase 4 calibration (`src/evaluation/generation/calibrate.py`), so scores stay interpretable.
- For option B or C, measure agreement with the existing four-call scores on a sample of traces first. The `judge_*` scores already stored in Langfuse are the baseline to compare against.
