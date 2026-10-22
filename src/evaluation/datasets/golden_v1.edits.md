# Edit log — golden_v1.reviewed.jsonl

Generated 2026-08-15T14:15:35+00:00 by claude-expert-review.

Every question below was changed. Each entry shows the original wording, the flaw, the
replacement, and why. The `leak` column re-runs the trigram/hapax filter on the *edited*
text, so an edit cannot reintroduce the leakage the filter was built to catch.

Read this file to check my judgement. Anything you disagree with, edit the question in
`golden_v1.reviewed.jsonl` directly — the row's `review.notes` records the same rationale.

---

## Coverage rescues

These three papers each had exactly one usable question left. Rejecting it would have dropped
the paper out of the evaluation set, leaving it in the index as a pure distractor — extra
retrieval difficulty with no chance of ever being a target. Edited rather than cut.

### q0006  (single_hop) — 2402.17345v1
- **Flaw:** Refers to the document: 'in this graph representation method'.
- **Before:** How does the training strategy for balancing contrastive and masking objectives evolve over time in this graph representation method?
- **After:**  How does the weighting between contrastive and masking objectives change over the course of training in graph self-supervised learning?
- **Why:** COVERAGE RESCUE - 2402.17345v1's only surviving question. Replaced the document reference with the subject area.
- **Leakage recheck:** 0.0% ok

### q0039  (single_hop) — 2606.19781v2
- **Flaw:** Leakage filter: 27.3% trigram overlap with the source chunk.
- **Before:** What are the computed scaling exponents for optimal model size and dataset size as functions of compute in the scratch training setup?
- **After:**  When training from scratch, what exponents describe how the best parameter count and training-set size grow with the compute budget?
- **Why:** COVERAGE RESCUE - 2606.19781v2 had zero surviving questions. Reworded 'computed scaling exponents for optimal model size and dataset size as functions of compute' into ordinary terminology. NOTE: source snippet is a flattened table row, but values are labelled (N*/D*, loss-minimum vs parametric) so the answer is attributable.
- **Leakage recheck:** 0.0% ok

### q0072  (single_hop) — 2508.06617v3
- **Flaw:** Refers to the document: 'the proposed scaling law'.
- **Before:** How does the proposed scaling law compare in prediction accuracy to existing scaling laws for sparse models, based on MSE?
- **After:**  How closely does a unified sparse-and-dense scaling law agree with Frantar's sparse-model law in mean squared error?
- **Why:** COVERAGE RESCUE - 2508.06617v3's only surviving question. Named the laws by what they model rather than by authorship of the paper.
- **Leakage recheck:** 0.0% ok

---

## Reworded

### q0002  (single_hop) — 2401.11864v5
- **Flaw:** 'Student models' assumes the reader already knows this is a distillation paper.
- **Before:** Which student models were fine-tuned and what were their parameter sizes?
- **After:**  In knowledge distillation for code tasks, which CodeT5 variants served as students and how many parameters did each have?
- **Why:** Added the task context and named the model family so the question stands alone.
- **Leakage recheck:** 0.0% ok

### q0014  (single_hop) — 2409.08062v1
- **Flaw:** 'The system' is vague.
- **Before:** How does the system select actions during inference when using the Q-value module?
- **After:**  How does a decision-transformer agent pick actions at inference time when a Q-value module scores candidate returns?
- **Why:** Named the agent type and the module's role.
- **Leakage recheck:** 0.0% ok

### q0017  (single_hop) — 2410.14961v1
- **Flaw:** Refers to the document: 'selected for evaluation in the study of'.
- **Before:** Which standard graph file formats were selected for evaluation in the study of LLM performance on graph textualization?
- **After:**  Which standard graph file formats are used when testing how well LLMs read textualized graphs?
- **Why:** Dropped 'the study'; the question now names only the subject.
- **Leakage recheck:** 0.0% ok

### q0030  (single_hop) — 2505.07884v1
- **Flaw:** 'In the system' is vague.
- **Before:** Which machine learning models were used for named entity recognition in the system?
- **After:**  Which classical and deep-learning models were combined for multilingual named entity recognition?
- **Why:** Replaced with the task, which is what distinguishes this paper.
- **Leakage recheck:** 0.0% ok

### q0035  (single_hop) — 2510.00261v1
- **Flaw:** Refers to the document: 'used as the primary approach in the experiments'.
- **Before:** Which ECG-to-text encoding method is used as the primary approach in the experiments and why?
- **After:**  Which method of turning ECG signals into text is chosen for being simple and cheap to run, and on what evidence?
- **Why:** Removed the document reference. First rewrite still tripped the leakage filter at 20% because 'low computational overhead' is lifted verbatim from the chunk; this wording paraphrases it and scores 0%.
- **Leakage recheck:** 0.0% ok

### q0053  (single_hop) — 2410.17635v2
- **Flaw:** Refers to the document: 'in this method'.
- **Before:** How many new questions were generated using GPT-4 in this method, and what does that replace compared to prior work?
- **After:**  How many new questions did GPT-4 generate for the training set, and what step does that replace relative to earlier distillation work?
- **Why:** Kept both halves of the question; removed the deictic reference.
- **Leakage recheck:** 0.0% ok

### q0067  (single_hop) — 2401.11864v5
- **Flaw:** 'The system' is vague.
- **Before:** How does the system decide which reasoning output to use when multiple types are generated?
- **After:**  When a model produces program, equation, and chain-of-thought answers, in what order are they tried to pick the final one?
- **Why:** Named the three output types and made the fallback ordering the explicit subject.
- **Leakage recheck:** 0.0% ok

### q0104  (multi_hop) — 2406.15962v1, 2412.00687v1
- **Flaw:** Refers to the documents: 'the systems described'.
- **Before:** Do the systems described use cryptographic watermarking or differential privacy to trace data leaks during medical data sharing?
- **After:**  Is digital watermarking or differential privacy used to trace who leaked data when medical records are shared?
- **Why:** Named the two mechanisms directly. Still needs both papers, so it stays multi-hop.
- **Leakage recheck:** 0.0% ok

### q0111  (multi_hop) — 2404.12267v1, 2601.07440v1
- **Flaw:** Refers to the documents: 'the methods described'.
- **Before:** Do the methods described for inferring posterior distributions rely on the same type of neural network architecture for encoding or decoding data?
- **After:**  Do variational autoencoders and normalizing-flow autoencoders use the same encoder-decoder structure when inferring posteriors?
- **Why:** Named both architectures explicitly, which is how a researcher would phrase it.
- **Leakage recheck:** 0.0% ok

### q0148  (comparative) — 2409.08062v1, 2410.14069v1
- **Flaw:** Presupposed that convolution reduces sequence-length dependency, which the reference answer denies; the comparison was left unresolved.
- **Before:** Does using a convolution module instead of a causal transformer reduce the sequence length dependency during training compared to optimizing policy via partial optimal transport?
- **After:**  How does a convolution-based sequence model differ from a partial-optimal-transport objective in which actions it learns to reproduce in offline RL?
- **Why:** Reframed onto a dimension both chunks actually support: which actions each approach learns to reproduce.
- **Leakage recheck:** 0.0% ok

### q0154  (unanswerable) — 
- **Flaw:** Named the Amazon product co-purchase dataset, but the corpus does contain Amazon recommendation data (ARAG uses Amazon Clothing/Electronics/Home), so it may be partly answerable.
- **Before:** What is the mean average precision of the graph neural network on the Amazon product co-purchase dataset?
- **After:**  What is the mean average precision of the graph neural network on the PDBbind protein-ligand binding benchmark?
- **Why:** Swapped to protein-ligand binding affinity. The corpus covers molecular property prediction but no protein-ligand work, so this stays a plausible near-miss that is genuinely unanswerable.
- **Leakage recheck:** 0.0% ok

### q0177  (paraphrase) — 2404.15817v1
- **Flaw:** 'Can you outline...' is a chat request, not a search query.
- **Before:** Can you outline the three main parts of a Domain Adversarial Neural Network and explain how each contributes to adapting across domains?
- **After:**  What are the three main parts of a Domain Adversarial Neural Network, and how does each contribute to cross-domain adaptation?
- **Why:** Converted to a direct question; meaning and ground truth unchanged.
- **Leakage recheck:** 12.5% ok

---

## Reclassified

Question text unchanged — the *label* was wrong. Both were filed as two-document questions
but one chunk contributed nothing, so the ground truth demanded a retrieval the question
never actually required.

### q0110  (multi_hop -> single_hop) — now 2502.07154v4
- **Flaw:** Labelled multi-hop, but snippet A ('We evaluate from two perspectives: reasoning efficiency and reasoning accuracy') supports nothing. Only paper B answers it.
- **Question (unchanged):** Does fine-tuning with the DCO objective reduce overconfidence in greedy completions more effectively than CE loss for larger language models on mathematical reasoning tasks?
- **Change:** reclassified multi_hop -> single_hop; dropped chunk(s) from 2410.17635v2
- **Why:** Reclassified to single_hop and dropped the 2410.17635v2 chunk, so the ground truth reflects what is actually needed.

### q0143  (comparative -> single_hop) — now 2510.00261v1
- **Flaw:** Labelled comparative, but snippet A is about RAG limitations in recommendation and contributes nothing; paper B answers it outright.
- **Question (unchanged):** Does using retrieval-augmented generation during both training and inference yield higher accuracy than using it only during inference?
- **Change:** reclassified comparative -> single_hop; dropped chunk(s) from 2506.21931v2
- **Why:** Reclassified to single_hop and dropped the 2506.21931v2 chunk. The question itself is one of the better ones in the set.

---
## Summary

- questions edited: **17** (3 coverage rescues, 12 rewordings, 2 reclassifications)
- leakage failures after editing: **none**
- papers covered by usable questions: **43/43**
