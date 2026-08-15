"""Topic definitions for the evaluation corpus.

Why topics rather than one broad query per category: a single
``cat:cs.AI`` query sorted by date returns whatever happened to be posted that
week, which clusters hard around a handful of trending subjects. Retrieval
metrics measured on such a corpus are biased — every query looks like every
document, so BM25 and vector search both appear to work well and neither is
distinguished.

Sampling a fixed quota from each of many named topics makes the spread a
property of the design instead of an accident of the calendar.

Quotas are set here so the totals are auditable at a glance:
cs.AI = 10 topics x 3 papers = 30, cs.LG = 10 topics x 2 papers = 20.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Topic:
    """One topical slice of the corpus.

    :param key: Stable identifier, recorded in the manifest so a paper can be
        traced back to the slice it was selected for.
    :param category: arXiv category this slice draws from (``cs.AI``/``cs.LG``).
    :param terms: Phrases matched against title and abstract, OR-ed together.
    :param quota: How many papers to keep from this slice.
    :param label: Human-readable description for reports.
    """

    key: str
    category: str
    terms: Tuple[str, ...]
    quota: int
    label: str


# --------------------------------------------------------------------------
# cs.AI — Gen AI focus (30 papers)
# --------------------------------------------------------------------------

CS_AI_TOPICS: Tuple[Topic, ...] = (
    Topic(
        key="llm_general",
        category="cs.AI",
        terms=("large language model", "foundation model"),
        quota=3,
        label="Large language models (general)",
    ),
    Topic(
        key="lora_peft",
        category="cs.AI",
        terms=("LoRA", "low-rank adaptation", "parameter-efficient fine-tuning", "PEFT"),
        quota=3,
        label="LoRA and parameter-efficient fine-tuning",
    ),
    Topic(
        key="open_weight_models",
        category="cs.AI",
        terms=("Llama", "Mistral", "Qwen", "open-weight model"),
        quota=3,
        label="Open-weight model families (Llama / Mistral / Qwen)",
    ),
    Topic(
        key="transformer_architecture",
        category="cs.AI",
        terms=("transformer architecture", "attention mechanism", "self-attention", "state space model"),
        quota=3,
        label="Transformer and attention architectures",
    ),
    Topic(
        key="pretraining_scaling",
        category="cs.AI",
        terms=("pretraining", "pre-training corpus", "scaling law", "training compute"),
        quota=3,
        label="LLM pretraining and scaling laws",
    ),
    Topic(
        key="agentic_ai",
        category="cs.AI",
        terms=("LLM agent", "agentic", "tool use", "multi-agent system"),
        quota=3,
        label="Agentic AI and tool-using LLMs",
    ),
    Topic(
        key="rag",
        category="cs.AI",
        terms=("retrieval-augmented generation", "retrieval augmented", "RAG pipeline", "knowledge grounding"),
        quota=3,
        label="Retrieval-augmented generation",
    ),
    Topic(
        key="rlhf_preference",
        category="cs.AI",
        terms=(
            "reinforcement learning from human feedback",
            "RLHF",
            "direct preference optimization",
            "reward model",
        ),
        quota=3,
        label="RLHF and preference optimization",
    ),
    Topic(
        key="instruction_alignment",
        category="cs.AI",
        terms=("instruction tuning", "alignment", "safety alignment", "hallucination"),
        quota=3,
        label="Instruction tuning, alignment, and hallucination",
    ),
    Topic(
        key="reasoning",
        category="cs.AI",
        terms=("chain-of-thought", "reasoning capability", "test-time compute", "mathematical reasoning"),
        quota=3,
        label="Reasoning and chain-of-thought",
    ),
)


# --------------------------------------------------------------------------
# cs.LG — deliberately broad (20 papers)
# --------------------------------------------------------------------------

CS_LG_TOPICS: Tuple[Topic, ...] = (
    Topic(
        key="computer_vision",
        category="cs.LG",
        terms=("image classification", "object detection", "vision transformer", "semantic segmentation"),
        quota=2,
        label="Computer vision",
    ),
    Topic(
        key="nlp",
        category="cs.LG",
        terms=("named entity recognition", "machine translation", "text classification", "sentiment analysis"),
        quota=2,
        label="Natural language processing",
    ),
    Topic(
        key="recommendation",
        category="cs.LG",
        terms=("recommender system", "collaborative filtering", "click-through rate", "personalization"),
        quota=2,
        label="Recommendation systems",
    ),
    Topic(
        key="time_series",
        category="cs.LG",
        terms=("time series forecasting", "anomaly detection", "temporal prediction"),
        quota=2,
        label="Time series and forecasting",
    ),
    Topic(
        key="graph_learning",
        category="cs.LG",
        terms=("graph neural network", "node classification", "link prediction", "graph representation"),
        quota=2,
        label="Graph machine learning",
    ),
    Topic(
        key="generative_models",
        category="cs.LG",
        terms=("diffusion model", "variational autoencoder", "normalizing flow", "generative adversarial"),
        quota=2,
        label="Generative models (diffusion / VAE / GAN)",
    ),
    Topic(
        key="optimization",
        category="cs.LG",
        terms=("stochastic gradient descent", "optimizer", "convergence rate", "learning rate schedule"),
        quota=2,
        label="Optimization and training dynamics",
    ),
    Topic(
        key="federated_privacy",
        category="cs.LG",
        terms=("federated learning", "differential privacy", "privacy-preserving"),
        quota=2,
        label="Federated learning and privacy",
    ),
    Topic(
        key="reinforcement_learning",
        category="cs.LG",
        terms=("reinforcement learning", "policy gradient", "offline reinforcement learning", "Markov decision process"),
        quota=2,
        label="Reinforcement learning and control",
    ),
    Topic(
        key="representation_learning",
        category="cs.LG",
        terms=("self-supervised learning", "contrastive learning", "representation learning", "transfer learning"),
        quota=2,
        label="Self-supervised and representation learning",
    ),
)


ALL_TOPICS: Tuple[Topic, ...] = CS_AI_TOPICS + CS_LG_TOPICS


def topics_for(category: str) -> Tuple[Topic, ...]:
    """Return the topic slices defined for a category.

    :param category: ``"cs.AI"`` or ``"cs.LG"``.
    :returns: The topics for that category.
    :raises ValueError: If the category has no topics defined.
    """
    matches = tuple(topic for topic in ALL_TOPICS if topic.category == category)
    if not matches:
        raise ValueError(f"No topics defined for category {category!r}")
    return matches


def quota_for(category: str) -> int:
    """Total paper quota across a category's topics.

    :param category: ``"cs.AI"`` or ``"cs.LG"``.
    :returns: Sum of the topic quotas.
    """
    return sum(topic.quota for topic in topics_for(category))


def validate_topics(topics: Sequence[Topic] = ALL_TOPICS) -> List[str]:
    """Check the topic table for the mistakes that break a corpus build.

    Run as a guard by the discover CLI, so a typo surfaces before spending
    minutes on rate-limited arXiv calls.

    :param topics: Topics to validate.
    :returns: Problem descriptions; empty when the table is sound.
    """
    problems: List[str] = []

    seen_keys = set()
    for topic in topics:
        if topic.key in seen_keys:
            problems.append(f"duplicate topic key: {topic.key}")
        seen_keys.add(topic.key)

        if not topic.terms:
            problems.append(f"{topic.key}: no search terms")
        if topic.quota < 1:
            problems.append(f"{topic.key}: quota must be >= 1, got {topic.quota}")
        for term in topic.terms:
            # A stray quote would silently corrupt the phrase query built from it.
            if '"' in term:
                problems.append(f"{topic.key}: term contains a quote character: {term!r}")
            if not term.strip():
                problems.append(f"{topic.key}: empty search term")

    return problems
