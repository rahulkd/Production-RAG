# Generation leaderboard

Appended by `python -m src.evaluation.generation.runner`; never edit or reorder rows.
Values are mean ± 95% CI. Faithfulness = supported claims / claims, over attempted answers; contradictions per answer
are reported separately. Relevance is 0-3 over answerable questions (a refusal scores 0). Citation precision = cited
papers that were retrieved / cited papers. Refusal recall is over unanswerable questions, false refusal over answerable.

| run_id | retrieval | judge | faithful | contra/ans | relevance | cite prec | cite present | refusal recall | false refusal | json ok | n | dataset |
|--------|-----------|-------|----------|------------|-----------|-----------|--------------|----------------|---------------|---------|---|---------|
| 20261024-083836 | hybrid k=10 | deepseek.v3.2 | 0.943 ±.028 | 0.080 | 2.733 ±.145 | 1.000 ±.000 | 0.400 | 1.000 ±.000 | 0.033 ±.037 | 0.982 | 110 | 79c42919 |
