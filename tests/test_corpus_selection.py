"""Tests for corpus selection: filtering, deduplication, quotas, backfill.

Deduplication is the load-bearing rule here. Gen-AI papers are routinely
cross-listed under both cs.AI and cs.LG, so without a shared claim set the
same paper would be selected twice and the corpus would look more diverse
than it is.
"""

import pytest
from src.evaluation.corpus.selection import Candidate, base_arxiv_id, select_corpus
from src.evaluation.corpus.topics import Topic

FROM_DATE = "20240101"
TO_DATE = "20260809"


def topic(key: str, quota: int = 2, category: str = "cs.AI") -> Topic:
    """Build a throwaway topic for a test.

    :param key: Topic key.
    :param quota: Papers wanted from this slice.
    :param category: arXiv category.
    :returns: The topic.
    """
    return Topic(key=key, category=category, terms=("term",), quota=quota, label=key)


class TestBaseArxivId:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2501.01234v1", "2501.01234"),
            ("2501.01234v12", "2501.01234"),
            ("2501.01234", "2501.01234"),
            ("  2501.01234v3  ", "2501.01234"),
        ],
    )
    def test_strips_version(self, raw, expected):
        assert base_arxiv_id(raw) == expected

    def test_leaves_internal_v_alone(self):
        """Old-style IDs contain letters; only a trailing vN is a version."""
        assert base_arxiv_id("cs/0501001") == "cs/0501001"


class TestFiltering:
    def test_wrong_category_dropped(self, make_candidate):
        """cat: matching on arXiv is permissive, so results need re-checking."""
        report = select_corpus(
            [topic("t", quota=2)],
            {"t": [make_candidate("2501.00001", categories=("cs.CV",))]},
            FROM_DATE,
            TO_DATE,
        )
        assert report.total == 0
        assert report.rejected["wrong_category"] == 1

    def test_cross_listed_paper_is_eligible(self, make_candidate):
        report = select_corpus(
            [topic("t", quota=1)],
            {"t": [make_candidate("2501.00001", categories=("cs.LG", "cs.AI"))]},
            FROM_DATE,
            TO_DATE,
        )
        assert report.total == 1

    def test_paper_before_window_dropped(self, make_candidate):
        report = select_corpus(
            [topic("t", quota=2)],
            {"t": [make_candidate("2301.00001", published_date="2023-11-02T00:00:00Z")]},
            FROM_DATE,
            TO_DATE,
        )
        assert report.total == 0
        assert report.rejected["out_of_date_window"] == 1

    def test_paper_after_window_dropped(self, make_candidate):
        report = select_corpus(
            [topic("t", quota=2)],
            {"t": [make_candidate("2701.00001", published_date="2027-01-05T00:00:00Z")]},
            FROM_DATE,
            TO_DATE,
        )
        assert report.rejected["out_of_date_window"] == 1

    def test_missing_date_dropped(self, make_candidate):
        report = select_corpus(
            [topic("t")],
            {"t": [make_candidate("2501.00001", published_date="")]},
            FROM_DATE,
            TO_DATE,
        )
        assert report.rejected["missing_date"] == 1

    def test_missing_title_dropped(self, make_candidate):
        report = select_corpus(
            [topic("t")],
            {"t": [make_candidate("2501.00001", title="   ")]},
            FROM_DATE,
            TO_DATE,
        )
        assert report.rejected["missing_title"] == 1


class TestQuotas:
    def test_quota_is_respected(self, make_candidate):
        candidates = [make_candidate(f"2501.0000{i}") for i in range(1, 6)]
        report = select_corpus([topic("t", quota=2)], {"t": candidates}, FROM_DATE, TO_DATE)
        assert report.per_topic["t"] == 2
        assert report.total == 2

    def test_order_is_preserved(self, make_candidate):
        candidates = [make_candidate("2501.00001"), make_candidate("2501.00002"), make_candidate("2501.00003")]
        report = select_corpus([topic("t", quota=2)], {"t": candidates}, FROM_DATE, TO_DATE)
        assert [s.candidate.arxiv_id for s in report.selected] == ["2501.00001", "2501.00002"]

    def test_shortfall_is_reported(self, make_candidate):
        report = select_corpus(
            [topic("t", quota=3)],
            {"t": [make_candidate("2501.00001")]},
            FROM_DATE,
            TO_DATE,
            target_total=3,
        )
        assert report.shortfalls["t"] == 2
        assert report.total == 1

    def test_missing_topic_key_is_tolerated(self):
        """A failed topic query yields no key at all; that must not crash."""
        report = select_corpus([topic("t", quota=2)], {}, FROM_DATE, TO_DATE)
        assert report.total == 0
        assert report.shortfalls["t"] == 2


class TestDeduplication:
    def test_same_paper_claimed_once_across_topics(self, make_candidate):
        shared = make_candidate("2501.00001")
        report = select_corpus(
            [topic("a", quota=1), topic("b", quota=1)],
            {"a": [shared], "b": [shared]},
            FROM_DATE,
            TO_DATE,
            target_total=2,
        )
        assert report.total == 1
        assert report.per_topic["a"] == 1
        assert report.per_topic["b"] == 0

    def test_dedup_ignores_version(self, make_candidate):
        report = select_corpus(
            [topic("a", quota=1), topic("b", quota=1)],
            {"a": [make_candidate("2501.00001v1")], "b": [make_candidate("2501.00001v2")]},
            FROM_DATE,
            TO_DATE,
            target_total=2,
        )
        assert report.total == 1

    def test_claimed_set_shared_between_categories(self, make_candidate):
        """A cross-listed paper taken by cs.AI must not be re-taken by cs.LG."""
        cross_listed = make_candidate("2501.00001", categories=("cs.AI", "cs.LG"))
        claimed: set = set()

        ai = select_corpus([topic("ai", quota=1)], {"ai": [cross_listed]}, FROM_DATE, TO_DATE, claimed=claimed)
        lg = select_corpus(
            [topic("lg", quota=1, category="cs.LG")],
            {"lg": [cross_listed]},
            FROM_DATE,
            TO_DATE,
            claimed=claimed,
        )

        assert ai.total == 1
        assert lg.total == 0
        assert claimed == {"2501.00001"}

    def test_duplicates_within_one_topic_collapse(self, make_candidate):
        report = select_corpus(
            [topic("t", quota=3)],
            {"t": [make_candidate("2501.00001"), make_candidate("2501.00001v2"), make_candidate("2501.00002")]},
            FROM_DATE,
            TO_DATE,
            target_total=3,
        )
        assert report.total == 2


class TestBackfill:
    def test_shortfall_is_backfilled_from_other_topics(self, make_candidate):
        """A topic returning nothing must not leave the category under target."""
        rich = [make_candidate(f"2501.1000{i}") for i in range(1, 6)]
        report = select_corpus(
            [topic("empty", quota=2), topic("rich", quota=2)],
            {"empty": [], "rich": rich},
            FROM_DATE,
            TO_DATE,
            target_total=4,
        )
        assert report.total == 4
        assert report.per_topic["rich"] == 4
        assert sum(1 for s in report.selected if s.via_backfill) == 2

    def test_backfill_is_round_robin(self, make_candidate):
        """Spread the deficit rather than draining one topic."""
        a = [make_candidate(f"2501.2000{i}") for i in range(1, 5)]
        b = [make_candidate(f"2501.3000{i}") for i in range(1, 5)]
        report = select_corpus(
            [topic("dead", quota=2), topic("a", quota=1), topic("b", quota=1)],
            {"dead": [], "a": a, "b": b},
            FROM_DATE,
            TO_DATE,
            target_total=4,
        )
        assert report.total == 4
        assert report.per_topic["a"] == 2
        assert report.per_topic["b"] == 2

    def test_backfill_stops_when_candidates_exhausted(self, make_candidate):
        report = select_corpus(
            [topic("t", quota=5)],
            {"t": [make_candidate("2501.00001")]},
            FROM_DATE,
            TO_DATE,
            target_total=5,
        )
        assert report.total == 1

    def test_no_backfill_when_target_met(self, make_candidate):
        candidates = [make_candidate(f"2501.0000{i}") for i in range(1, 5)]
        report = select_corpus([topic("t", quota=2)], {"t": candidates}, FROM_DATE, TO_DATE, target_total=2)
        assert report.total == 2
        assert not any(s.via_backfill for s in report.selected)


class TestReport:
    def test_primary_category_counts(self, make_candidate):
        report = select_corpus(
            [topic("t", quota=3)],
            {
                "t": [
                    make_candidate("2501.00001", categories=("cs.AI", "cs.LG")),
                    make_candidate("2501.00002", categories=("cs.LG", "cs.AI")),
                    make_candidate("2501.00003", categories=("cs.AI",)),
                ]
            },
            FROM_DATE,
            TO_DATE,
        )
        assert report.primary_category_counts() == {"cs.AI": 2, "cs.LG": 1}

    def test_primary_category_is_first_listed(self, make_candidate):
        candidate = make_candidate("2501.00001", categories=("cs.LG", "cs.AI"))
        assert candidate.primary_category == "cs.LG"


class TestGuards:
    def test_empty_topics_rejected(self):
        with pytest.raises(ValueError, match="at least one topic"):
            select_corpus([], {}, FROM_DATE, TO_DATE)

    def test_mixed_categories_rejected(self):
        with pytest.raises(ValueError, match="share one category"):
            select_corpus(
                [topic("a", category="cs.AI"), topic("b", category="cs.LG")],
                {},
                FROM_DATE,
                TO_DATE,
            )

    def test_candidate_without_categories_has_empty_primary(self):
        candidate = Candidate(
            arxiv_id="2501.00001",
            title="T",
            abstract="A",
            authors=[],
            categories=[],
            published_date="2025-01-01T00:00:00Z",
        )
        assert candidate.primary_category == ""
