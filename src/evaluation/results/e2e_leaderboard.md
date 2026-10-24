# End-to-end leaderboard

Appended by `python -m src.evaluation.e2e.runner`; never edit or reorder rows.
Each row is the golden set sent over HTTP to one route at one top_k. `agree` = share of responses whose source papers
equal the in-process harness retrieval for the same query (Q11). Quality columns use the Phase 4 judge on the served answer;
faithfulness is scored only where the papers agree. Latency is wall-clock per request; TTFT applies to /stream only.

| run_id | route | k | agree | hybrid | faithful | relevance | refusal recall | false refusal | cite present | p50 s | p95 s | TTFT p50 s | errors | dataset |
|--------|-------|---|-------|--------|----------|-----------|----------------|---------------|--------------|-------|-------|------------|--------|---------|
| 20261024-115413 | /ask | 3 | 1.000 | 110/110 | 0.939 ±.036 | 2.300 ±.224 | 0.950 | 0.122 | 0.309 | 1.9 | 2.5 | – | 0 | 79c42919 |
| 20261024-115413 | /stream | 3 | 1.000 | 110/110 | 0.960 ±.020 | 2.433 ±.217 | 0.950 | 0.089 | 0.973 | 1.9 | 3.0 | 1.3 | 0 | 79c42919 |
| 20261024-115413 | /ask | 10 | 1.000 | 110/110 | 0.946 ±.031 | 2.667 ±.155 | 1.000 | 0.033 | 0.418 | 2.4 | 3.8 | – | 0 | 79c42919 |
| 20261024-115413 | /stream | 10 | 1.000 | 110/110 | 0.952 ±.022 | 2.656 ±.155 | 0.950 | 0.022 | 0.864 | 2.5 | 3.4 | 1.7 | 0 | 79c42919 |
| 20261024-133958 | /ask | 3 | 1.000 | 110/110 | 0.945 ±.039 | 2.333 ±.221 | 1.000 | 0.111 | 0.364 | 2.0 | 3.7 | – | 0 | 79c42919 |
| 20261024-133958 | /stream | 3 | 1.000 | 110/110 | 0.942 ±.030 | 2.356 ±.224 | 0.950 | 0.100 | 1.000 | 1.9 | 2.6 | 1.4 | 0 | 79c42919 |
| 20261024-133958 | /ask | 10 | 1.000 | 110/110 | 0.968 ±.018 | 2.678 ±.154 | 1.000 | 0.022 | 0.418 | 2.3 | 3.1 | – | 0 | 79c42919 |
| 20261024-133958 | /stream | 10 | 1.000 | 110/110 | 0.949 ±.023 | 2.700 ±.153 | 0.950 | 0.022 | 0.882 | 2.2 | 3.0 | 1.6 | 0 | 79c42919 |
