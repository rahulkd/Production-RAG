"""Tests for the corpus manifest and its freeze guarantee.

The hash check is the mechanism that stops a changed parser or chunker from
silently invalidating Phase 2's ground-truth chunk labels, so it gets the most
coverage here.
"""

import json

import pytest
from src.evaluation.corpus.manifest import (
    MANIFEST_VERSION,
    CorpusManifest,
    ManifestPaper,
    text_sha256,
    verify_text_hashes,
)
from src.evaluation.logging_setup import FailureRecorder


def paper(base_id: str, sha: str | None = None, category: str = "cs.AI", topic_key: str = "rag") -> ManifestPaper:
    """Build a manifest entry for a test.

    :param base_id: Version-stripped arXiv ID.
    :param sha: Recorded text hash, or None for an unbuilt entry.
    :param category: Primary category.
    :param topic_key: Owning topic slice.
    :returns: The entry.
    """
    return ManifestPaper(
        arxiv_id=f"{base_id}v1",
        base_id=base_id,
        title=f"Paper {base_id}",
        primary_category=category,
        categories=[category],
        published_date="2025-03-14T00:00:00Z",
        topic_key=topic_key,
        raw_text_sha256=sha,
    )


class TestHashing:
    def test_deterministic(self):
        assert text_sha256("hello") == text_sha256("hello")

    def test_sensitive_to_whitespace(self):
        """Chunk boundaries shift on whitespace, so it must count as a change."""
        assert text_sha256("a b") != text_sha256("a  b")

    def test_hex_digest_shape(self):
        digest = text_sha256("x")
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


class TestRoundTrip:
    def test_save_load_preserves_everything(self, tmp_path):
        manifest = CorpusManifest(
            created_at="2026-08-09T10:00:00+00:00",
            date_window={"from": "20240101", "to": "20260809"},
            index_name="arxiv_evaluation_index",
            embedding_model="jina-embeddings-v3",
            embedding_dim=1024,
            chunking={"chunk_size": 600, "overlap_size": 100, "min_chunk_size": 100},
            papers=[paper("2501.00001", sha="a" * 64), paper("2501.00002")],
        )
        path = tmp_path / "manifest.json"
        manifest.save(path)

        loaded = CorpusManifest.load(path)
        assert loaded.index_name == "arxiv_evaluation_index"
        assert loaded.embedding_dim == 1024
        assert loaded.chunking["chunk_size"] == 600
        assert loaded.arxiv_ids == ["2501.00001v1", "2501.00002v1"]
        assert loaded.papers[0].raw_text_sha256 == "a" * 64
        assert loaded.papers[1].raw_text_sha256 is None

    def test_written_file_is_valid_json(self, tmp_path):
        path = tmp_path / "m.json"
        CorpusManifest(papers=[paper("2501.00001")]).save(path)
        payload = json.loads(path.read_text())
        assert payload["manifest_version"] == MANIFEST_VERSION
        assert len(payload["papers"]) == 1

    def test_missing_file_gives_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="discover"):
            CorpusManifest.load(tmp_path / "nope.json")

    def test_wrong_schema_version_rejected(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"manifest_version": 99, "papers": []}))
        with pytest.raises(ValueError, match="Unsupported manifest_version"):
            CorpusManifest.load(path)


class TestSummaries:
    def test_category_counts(self):
        manifest = CorpusManifest(
            papers=[paper("1", category="cs.AI"), paper("2", category="cs.AI"), paper("3", category="cs.LG")]
        )
        assert manifest.category_counts() == {"cs.AI": 2, "cs.LG": 1}

    def test_topic_counts(self):
        manifest = CorpusManifest(papers=[paper("1", topic_key="rag"), paper("2", topic_key="lora")])
        assert manifest.topic_counts() == {"rag": 1, "lora": 1}

    def test_by_base_id(self):
        manifest = CorpusManifest(papers=[paper("2501.00001")])
        assert "2501.00001" in manifest.by_base_id()

    def test_is_built_requires_every_paper(self):
        assert CorpusManifest(papers=[paper("1", sha="a" * 64)]).is_built
        assert not CorpusManifest(papers=[paper("1", sha="a" * 64), paper("2")]).is_built

    def test_empty_manifest_is_not_built(self):
        assert not CorpusManifest(papers=[]).is_built


class TestFreezeGuarantee:
    def test_unchanged_text_passes(self):
        text = "the extracted paper text"
        manifest = CorpusManifest(papers=[paper("2501.00001", sha=text_sha256(text))])
        assert verify_text_hashes(manifest, {"2501.00001": text}) == []

    def test_changed_text_is_caught(self):
        """A new arXiv version or docling upgrade must fail loudly."""
        manifest = CorpusManifest(papers=[paper("2501.00001", sha=text_sha256("original"))])
        failures = verify_text_hashes(manifest, {"2501.00001": "re-extracted, subtly different"})
        assert len(failures) == 1
        assert failures[0].arxiv_id == "2501.00001v1"
        assert not failures[0].matches

    def test_unbuilt_papers_are_skipped(self):
        """First build has no recorded hash; that is not a mismatch."""
        manifest = CorpusManifest(papers=[paper("2501.00001", sha=None)])
        assert verify_text_hashes(manifest, {"2501.00001": "anything"}) == []

    def test_papers_absent_from_this_run_are_skipped(self):
        manifest = CorpusManifest(papers=[paper("2501.00001", sha="a" * 64)])
        assert verify_text_hashes(manifest, {}) == []

    def test_only_the_changed_paper_is_reported(self):
        stable, drifted = "stable text", "was this"
        manifest = CorpusManifest(
            papers=[
                paper("2501.00001", sha=text_sha256(stable)),
                paper("2501.00002", sha=text_sha256(drifted)),
            ]
        )
        failures = verify_text_hashes(manifest, {"2501.00001": stable, "2501.00002": "now this"})
        assert [f.arxiv_id for f in failures] == ["2501.00002v1"]


class TestFailureRecorder:
    def test_records_and_groups_by_stage(self):
        recorder = FailureRecorder()
        recorder.record("2501.00001v1", "parse", "docling returned nothing")
        recorder.record("2501.00002v1", "parse", "docling returned nothing")
        recorder.record("2501.00003v1", "download", "404")

        assert len(recorder) == 3
        grouped = recorder.by_stage()
        assert len(grouped["parse"]) == 2
        assert len(grouped["download"]) == 1

    def test_report_lists_ids_for_copy_paste(self, tmp_path):
        recorder = FailureRecorder()
        recorder.record("2501.00001v1", "parse", "docling returned nothing", title="A Paper")

        text_path = tmp_path / "failures.log"
        recorder.write(text_path, tmp_path / "failures.json")

        report = text_path.read_text()
        assert "2501.00001v1" in report
        assert "arxiv_ids: 2501.00001v1" in report
        assert "A Paper" in report

    def test_clean_run_still_writes_a_report(self, tmp_path):
        """An empty file is evidence; a missing file is ambiguous."""
        text_path = tmp_path / "failures.log"
        FailureRecorder().write(text_path)
        assert text_path.exists()
        assert "No failures recorded" in text_path.read_text()

    def test_json_report_is_machine_readable(self, tmp_path):
        recorder = FailureRecorder()
        recorder.record("2501.00001v1", "empty_text", "12 chars")

        json_path = tmp_path / "failures.json"
        recorder.write(tmp_path / "failures.log", json_path)

        payload = json.loads(json_path.read_text())
        assert payload["failure_count"] == 1
        assert payload["by_stage"]["empty_text"] == ["2501.00001v1"]
        assert payload["failures"][0]["reason"] == "12 chars"
