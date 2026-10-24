"""Tests for the production/evaluation index switch and the evaluation freeze."""

import pytest
from pydantic import ValidationError
from src.config import OpenSearchSettings, Settings
from src.evaluation.corpus.freeze import WRITE_BLOCK, freeze, is_frozen, lift, verify_against_manifest
from src.evaluation.corpus.manifest import CorpusManifest, ManifestPaper
from src.services.opensearch import factory
from src.services.opensearch.client import OpenSearchClient


def make_settings(monkeypatch, **env):
    for key in ("OPENSEARCH__ACTIVE_INDEX", "OPENSEARCH__INDEX_NAME", "OPENSEARCH__EVALUATION_INDEX_NAME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(opensearch=OpenSearchSettings(_env_file=None))


# --- the switch ---------------------------------------------------------------


def test_defaults_to_production(monkeypatch):
    opensearch = make_settings(monkeypatch).opensearch
    assert opensearch.active_index == "production"
    assert opensearch.search_index_name == "arxiv-papers-chunks"


def test_evaluation_selects_the_frozen_index(monkeypatch):
    opensearch = make_settings(monkeypatch, OPENSEARCH__ACTIVE_INDEX="evaluation").opensearch
    assert opensearch.search_index_name == "arxiv_evaluation_index"
    # The production name is unaffected, so writers can still find it.
    assert opensearch.production_chunk_index == "arxiv-papers-chunks"


def test_evaluation_index_name_is_a_full_name_not_suffixed(monkeypatch):
    opensearch = make_settings(monkeypatch, OPENSEARCH__ACTIVE_INDEX="evaluation", OPENSEARCH__EVALUATION_INDEX_NAME="eval_v2").opensearch
    assert opensearch.search_index_name == "eval_v2"


def test_unknown_value_fails_at_startup_rather_than_falling_back(monkeypatch):
    with pytest.raises(ValidationError):
        make_settings(monkeypatch, OPENSEARCH__ACTIVE_INDEX="eval")


def test_production_name_follows_index_name_and_suffix(monkeypatch):
    assert make_settings(monkeypatch, OPENSEARCH__INDEX_NAME="papers-v2").opensearch.production_chunk_index == "papers-v2-chunks"


# --- search reads the active index, ingestion never does ----------------------


def test_search_client_follows_the_switch_but_writers_stay_on_production(monkeypatch):
    settings = make_settings(monkeypatch, OPENSEARCH__ACTIVE_INDEX="evaluation")
    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    factory.make_search_client.cache_clear()
    try:
        assert factory.make_search_client().index_name == "arxiv_evaluation_index"
        # Airflow's indexer and verify task build their client like this.
        assert factory.make_opensearch_client_fresh(settings).index_name == "arxiv-papers-chunks"
    finally:
        factory.make_search_client.cache_clear()


def test_explicit_index_override_still_wins(monkeypatch):
    settings = make_settings(monkeypatch)
    assert OpenSearchClient(host="http://x:9200", settings=settings, index_name="custom").index_name == "custom"


# --- freeze -------------------------------------------------------------------


class FakeIndices:
    def __init__(self, exists=True, blocked=None):
        self.present, self.blocked, self.puts = exists, blocked, []

    def exists(self, index):
        return self.present

    def get_settings(self, index, flat_settings=True):
        return {index: {"settings": {} if self.blocked is None else {WRITE_BLOCK: self.blocked}}}

    def refresh(self, index):
        pass

    def put_settings(self, index, body):
        self.puts.append(body)
        self.blocked = str(body[WRITE_BLOCK]).lower()


class FakeOpenSearch:
    def __init__(self, chunks, papers, **kwargs):
        self.indices, self._chunks, self._papers = FakeIndices(**kwargs), chunks, papers

    def count(self, index):
        return {"count": self._chunks}

    def search(self, index, body):
        return {"aggregations": {"p": {"value": self._papers}}}


def paper(n_chunks, base="2401.00001"):
    return ManifestPaper(
        arxiv_id=base + "v1", base_id=base, title="t", primary_category="cs.AI", categories=["cs.AI"],
        published_date="2024-01-01", topic_key="llm", raw_text_sha256="x" if n_chunks else None, n_chunks=n_chunks,
    )


def manifest(*papers, built_at="2026-01-01"):
    return CorpusManifest(
        manifest_version=1, created_at="", built_at=built_at, date_window={}, index_name="eval",
        embedding_model="m", embedding_dim=1024, chunking={}, papers=list(papers),
    )


def test_is_frozen_reads_the_write_block():
    assert not is_frozen(FakeOpenSearch(0, 0), "i")
    assert not is_frozen(FakeOpenSearch(0, 0, blocked="false"), "i")
    assert is_frozen(FakeOpenSearch(0, 0, blocked="true"), "i")
    assert not is_frozen(FakeOpenSearch(0, 0, exists=False), "i")


def test_freeze_blocks_writes_when_the_index_matches_the_manifest():
    client = FakeOpenSearch(chunks=30, papers=2)
    # A paper that failed to build has no chunks and is not expected in the index.
    counts = freeze(client, "i", manifest(paper(10, "2401.00001"), paper(20, "2401.00002"), paper(None, "2401.00003")))
    assert counts == {"chunks": 30, "papers": 2}
    assert is_frozen(client, "i")


@pytest.mark.parametrize("chunks, papers", [(29, 2), (31, 2), (30, 1), (30, 3)])
def test_freeze_refuses_an_index_that_differs_from_the_manifest(chunks, papers):
    client = FakeOpenSearch(chunks, papers)
    with pytest.raises(RuntimeError, match="Not freezing"):
        freeze(client, "i", manifest(paper(10, "2401.00001"), paper(20, "2401.00002")))
    assert client.indices.puts == []


def test_freeze_refuses_an_unbuilt_manifest_or_missing_index():
    for unbuilt in (manifest(paper(None)), manifest(paper(5), built_at=None)):
        with pytest.raises(RuntimeError, match="no completed build"):
            verify_against_manifest(FakeOpenSearch(0, 0), "i", unbuilt)
    with pytest.raises(RuntimeError, match="does not exist"):
        verify_against_manifest(FakeOpenSearch(0, 0, exists=False), "i", manifest(paper(5)))


def test_lift_unblocks_writes():
    client = FakeOpenSearch(5, 1, blocked="true")
    lift(client, "i")
    assert not is_frozen(client, "i")


def test_the_real_manifest_passes_the_freeze_precondition():
    # Regression: manifest.is_built is False for the real corpus (7 papers failed
    # to build), so the precondition must not depend on it.
    from src.evaluation import config as eval_config

    real = CorpusManifest.load(eval_config.CORPUS_MANIFEST)
    assert not real.is_built
    chunks = sum(p.n_chunks or 0 for p in real.papers)
    papers = sum(1 for p in real.papers if p.n_chunks)
    assert verify_against_manifest(FakeOpenSearch(chunks, papers), "i", real) == {"chunks": chunks, "papers": papers}
