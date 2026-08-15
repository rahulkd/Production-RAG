"""Tests for arXiv query construction.

The query string is the one part of discovery that cannot be validated
locally at runtime — a malformed query returns zero results rather than an
error, so a typo would silently produce an empty corpus slice.
"""

import pytest
from src.evaluation.corpus.query import (
    build_date_clause,
    build_terms_clause,
    build_topic_query,
    today_stamp,
)
from src.evaluation.corpus.topics import (
    ALL_TOPICS,
    CS_AI_TOPICS,
    CS_LG_TOPICS,
    Topic,
    quota_for,
    topics_for,
    validate_topics,
)


class TestDateClause:
    def test_expected_form(self):
        assert build_date_clause("20240101", "20260809") == "submittedDate:[202401010000 TO 202608092359]"

    def test_same_day_range_is_valid(self):
        assert build_date_clause("20240101", "20240101") == "submittedDate:[202401010000 TO 202401012359]"

    @pytest.mark.parametrize("bad", ["2024-01-01", "202401", "abcdefgh", "2024010", "202401011"])
    def test_malformed_date_rejected(self, bad):
        with pytest.raises(ValueError, match="YYYYMMDD"):
            build_date_clause(bad, "20260809")

    def test_impossible_date_rejected(self):
        with pytest.raises(ValueError, match="not a valid date"):
            build_date_clause("20240230", "20260809")

    def test_inverted_range_rejected(self):
        with pytest.raises(ValueError, match="after"):
            build_date_clause("20260809", "20240101")


class TestTermsClause:
    def test_phrases_are_quoted(self):
        """Unquoted multi-word terms would parse as separate bare tokens."""
        clause = build_terms_clause(["large language model"])
        assert clause == '(ti:"large language model" OR abs:"large language model")'

    def test_every_term_crossed_with_every_field(self):
        clause = build_terms_clause(["LoRA", "PEFT"], fields=["ti", "abs"])
        assert clause.count(" OR ") == 3
        for expected in ('ti:"LoRA"', 'abs:"LoRA"', 'ti:"PEFT"', 'abs:"PEFT"'):
            assert expected in clause

    def test_clause_is_parenthesised(self):
        """Without parens the OR would bind loosely against the AND terms."""
        clause = build_terms_clause(["RAG"])
        assert clause.startswith("(") and clause.endswith(")")

    def test_embedded_quote_rejected(self):
        with pytest.raises(ValueError, match="quote character"):
            build_terms_clause(['say "hi"'])

    def test_empty_term_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            build_terms_clause(["   "])

    def test_no_terms_rejected(self):
        with pytest.raises(ValueError, match="at least one search term"):
            build_terms_clause([])

    def test_no_fields_rejected(self):
        with pytest.raises(ValueError, match="at least one search field"):
            build_terms_clause(["RAG"], fields=[])


class TestTopicQuery:
    def test_full_query_shape(self):
        topic = Topic(key="t", category="cs.AI", terms=("RAG",), quota=3, label="RAG")
        query = build_topic_query(topic, "20240101", "20260809")
        assert query == ('cat:cs.AI AND (ti:"RAG" OR abs:"RAG") AND submittedDate:[202401010000 TO 202608092359]')

    def test_clauses_joined_with_and(self):
        query = build_topic_query(CS_AI_TOPICS[0], "20240101", "20260809")
        assert query.count(" AND ") >= 2
        assert query.startswith("cat:cs.AI AND (")

    @pytest.mark.parametrize("topic", ALL_TOPICS, ids=lambda t: t.key)
    def test_every_defined_topic_builds(self, topic):
        """A bad term in the topic table must fail here, not at fetch time."""
        query = build_topic_query(topic, "20240101", "20260809")
        assert f"cat:{topic.category}" in query
        assert query.count('"') % 2 == 0, "unbalanced quotes would corrupt the query"

    def test_today_stamp_format(self):
        stamp = today_stamp()
        assert len(stamp) == 8 and stamp.isdigit()


class TestTopicTable:
    def test_table_is_valid(self):
        assert validate_topics() == []

    def test_quotas_match_the_stated_targets(self):
        """30 cs.AI + 20 cs.LG is the requirement; keep the table honest about it."""
        assert quota_for("cs.AI") == 30
        assert quota_for("cs.LG") == 20

    def test_topic_counts(self):
        assert len(CS_AI_TOPICS) == 10
        assert len(CS_LG_TOPICS) == 10

    def test_keys_are_unique(self):
        keys = [topic.key for topic in ALL_TOPICS]
        assert len(keys) == len(set(keys))

    def test_topics_for_filters_by_category(self):
        assert all(t.category == "cs.LG" for t in topics_for("cs.LG"))

    def test_unknown_category_rejected(self):
        with pytest.raises(ValueError, match="No topics defined"):
            topics_for("cs.CV")

    def test_validator_catches_duplicate_keys(self):
        dupes = [
            Topic(key="x", category="cs.AI", terms=("a",), quota=1, label="x"),
            Topic(key="x", category="cs.AI", terms=("b",), quota=1, label="x"),
        ]
        assert any("duplicate" in problem for problem in validate_topics(dupes))

    def test_validator_catches_bad_quota_and_terms(self):
        bad = [Topic(key="y", category="cs.AI", terms=(), quota=0, label="y")]
        problems = validate_topics(bad)
        assert any("no search terms" in p for p in problems)
        assert any("quota" in p for p in problems)
