"""Tests for the golden-dataset pure logic.

The generation CLI needs OpenSearch and Bedrock, so it is not exercised here.
What is testable in isolation is the machinery that decides *what the model
sees* and *what counts as ground truth* — the two places where a silent bug
would produce a plausible-looking dataset that measures the wrong thing.
"""

import pytest
from src.evaluation.dataset.chunks import Chunk, eligible_chunks, group_by_paper, strip_header
from src.evaluation.dataset.generate import is_near_duplicate, parse_json_response, salvage_objects
from src.evaluation.dataset.leakage import (
    build_hapax_vocabulary,
    check_leakage,
    is_identifier_like,
    tokenize,
    trigram_overlap,
    trigrams,
)
from src.evaluation.dataset.sample import _spread_indices, sample_single_hop
from src.evaluation.dataset.schema import (
    GRADE_ANSWER,
    GRADE_SUPPORTING,
    GoldenQuestion,
    RelevantChunk,
    read_jsonl,
    validate_dataset,
    write_jsonl,
)

TITLE = "LangGFM: A Graph Foundation Model"
ABSTRACT = "Graph foundation models have gained attention. We propose a benchmark."


def make_chunk(index: int, body: str = "x " * 500, section: str = "3 Method", arxiv_id: str = "2501.00001v1") -> Chunk:
    """Build a Chunk with a realistic indexed form.

    :param index: Chunk index.
    :param body: Body text after the header.
    :param section: Section title.
    :param arxiv_id: Owning paper.
    :returns: The chunk.
    """
    return Chunk(
        arxiv_id=arxiv_id,
        chunk_index=index,
        section_title=section,
        body=body,
        full_text=f"{TITLE}\n\nAbstract: {ABSTRACT}\n\n{body}",
        title=TITLE,
        abstract=ABSTRACT,
    )


class TestStripHeader:
    def test_removes_exact_header(self):
        """The chunker prepends title+abstract to every chunk; questions must
        never be generated from it."""
        text = f"{TITLE}\n\nAbstract: {ABSTRACT}\n\nThe method uses a critic network."
        assert strip_header(text, TITLE, ABSTRACT) == "The method uses a critic network."

    def test_removes_leading_section_markers(self):
        text = f"{TITLE}\n\nAbstract: {ABSTRACT}\n\nSection: 3 Method\n\nWe train with Adam."
        assert strip_header(text, TITLE, ABSTRACT) == "We train with Adam."

    def test_removes_several_stacked_markers(self):
        text = f"{TITLE}\n\nAbstract: {ABSTRACT}\n\nSection: A\n\nSection: B\n\nBody here."
        assert strip_header(text, TITLE, ABSTRACT) == "Body here."

    def test_abstract_never_survives(self):
        """The whole point: no abstract text reaches the question generator."""
        text = f"{TITLE}\n\nAbstract: {ABSTRACT}\n\nUnrelated body text."
        assert ABSTRACT not in strip_header(text, TITLE, ABSTRACT)

    def test_falls_back_when_header_does_not_match(self):
        """A chunker change must degrade, not crash — but it is logged."""
        text = f"Different Title\n\nAbstract: {ABSTRACT}\n\nBody text."
        assert "Body text." in strip_header(text, TITLE, ABSTRACT)

    def test_body_containing_the_word_abstract_is_safe(self):
        text = f"{TITLE}\n\nAbstract: {ABSTRACT}\n\nWe abstract away the details."
        assert strip_header(text, TITLE, ABSTRACT) == "We abstract away the details."


class TestEligibility:
    def test_chunk_zero_excluded(self):
        """Chunk 0 is the title page and author list."""
        assert not make_chunk(0).is_eligible

    @pytest.mark.parametrize(
        "section",
        ["References (Part 1)", "REFERENCES", "Acknowledgments", "CCS CONCEPTS + KEYWORDS", "Appendix A", "Bibliography"],
    )
    def test_metadata_sections_excluded(self, section):
        assert not make_chunk(5, section=section).is_eligible

    def test_short_body_excluded(self):
        """A 200-char chunk cannot contain a substantive claim."""
        assert not make_chunk(5, body="short").is_eligible

    def test_real_content_included(self):
        assert make_chunk(5, body="y " * 600, section="4 Experiments").is_eligible

    def test_section_match_is_case_insensitive(self):
        assert not make_chunk(5, section="references").is_eligible

    def test_filter_reports_survivors(self):
        chunks = [make_chunk(0), make_chunk(1, section="References"), make_chunk(2)]
        assert [c.chunk_index for c in eligible_chunks(chunks)] == [2]


class TestLeakage:
    def test_verbatim_reuse_is_caught(self):
        source = "We stabilize the critic by adding an L2 penalty on target Q-values."
        assert trigram_overlap("How do we stabilize the critic by adding an L2 penalty?", source) > 0.15

    def test_paraphrase_passes(self):
        source = "We stabilize the critic by adding an L2 penalty on the target Q-values."
        assert trigram_overlap("Which regularization term keeps value estimates from diverging?", source) <= 0.15

    def test_shared_vocabulary_alone_is_not_leakage(self):
        """A question about LoRA must be allowed to say 'LoRA'."""
        source = "LoRA reduces trainable parameters by factorizing weight updates into low-rank matrices."
        assert trigram_overlap("How much memory does LoRA save?", source) <= 0.15

    def test_stopwords_are_ignored(self):
        assert "the" not in tokenize("the model and the data")

    def test_trigrams_of_short_input(self):
        assert trigrams(["a", "b"]) == [("a", "b")]
        assert trigrams([]) == []

    def test_empty_question_scores_zero(self):
        assert trigram_overlap("", "some source text here") == 0.0

    def test_hapax_vocabulary_finds_unique_identifiers(self):
        """Only identifier-like corpus-unique tokens count; see
        TestHapaxIsIdentifierLike for why ordinary words are excluded."""
        chunks = ["shared term appears here", "shared term again", "trained on qwen1 once"]
        hapax = build_hapax_vocabulary(chunks)
        assert "qwen1" in hapax
        assert "shared" not in hapax

    def test_hapax_reuse_is_rejected_even_at_low_overlap(self):
        """A single corpus-unique term makes a question trivially findable."""
        hapax = {"zzzunique"}
        verdict = check_leakage("What is zzzunique used for?", "unrelated source", hapax, 0.15)
        assert verdict.leaks
        assert "zzzunique" in verdict.hapax_hits

    def test_clean_question_passes_both_checks(self):
        verdict = check_leakage("How is the value function regularized?", "Totally different wording.", set(), 0.15)
        assert not verdict.leaks
        assert "clean" in verdict.reason


class TestSampling:
    def test_spread_picks_middle_for_one(self):
        """One question per paper should come from the body, not the intro."""
        assert _spread_indices(10, 1) == [5]

    def test_spread_covers_both_ends(self):
        """First, middle, last — step is (n-1)/(k-1) = 4.5, and round() uses
        banker's rounding, so the midpoint lands on 4 rather than 5."""
        assert _spread_indices(10, 3) == [0, 4, 9]

    def test_spread_is_monotonic_and_in_range(self):
        for available in (4, 7, 12, 39):
            for wanted in (1, 2, 3, 5):
                positions = _spread_indices(available, wanted)
                assert positions == sorted(positions)
                assert all(0 <= p < available for p in positions)

    def test_spread_handles_wanting_more_than_available(self):
        assert _spread_indices(3, 10) == [0, 1, 2]

    def test_spread_handles_zero(self):
        assert _spread_indices(5, 0) == []

    def test_every_paper_covered_before_any_repeats(self):
        """The load-bearing guarantee: an uncovered paper is a pure distractor."""
        chunks = [make_chunk(i, arxiv_id=f"250{p}.00001v1") for p in range(5) for i in range(1, 6)]
        seeds = sample_single_hop(chunks, target=5)
        assert len({chunk.arxiv_id for chunk in seeds}) == 5

    def test_extra_slots_go_to_richer_papers(self):
        chunks = [make_chunk(i, arxiv_id="2501.00001v1") for i in range(1, 11)]
        chunks += [make_chunk(i, arxiv_id="2502.00002v1") for i in range(1, 3)]
        seeds = sample_single_hop(chunks, target=6)
        assert len({c.arxiv_id for c in seeds}) == 2
        assert sum(1 for c in seeds if c.arxiv_id == "2501.00001v1") > 2

    def test_no_chunk_sampled_twice(self):
        chunks = [make_chunk(i, arxiv_id="2501.00001v1") for i in range(1, 11)]
        seeds = sample_single_hop(chunks, target=8)
        assert len({c.key for c in seeds}) == len(seeds)

    def test_group_by_paper_sorts_by_index(self):
        chunks = [make_chunk(5), make_chunk(1), make_chunk(3)]
        assert [c.chunk_index for c in group_by_paper(chunks)["2501.00001v1"]] == [1, 3, 5]


def make_question(qid: str = "q0001", **overrides) -> GoldenQuestion:
    """Build a valid question row for a test.

    :param qid: Question id.
    :param overrides: Field overrides.
    :returns: The question.
    """
    defaults = dict(
        qid=qid,
        question="Which regularization term stabilizes the critic?",
        question_type="single_hop",
        answerable=True,
        relevant_chunks=[RelevantChunk("2501.00001v1", 7, GRADE_ANSWER, chunk_text="body text here")],
        relevant_papers=["2501.00001v1"],
        reference_answer="An L2 penalty.",
    )
    defaults.update(overrides)
    return GoldenQuestion(**defaults)


class TestSchema:
    def test_valid_row_has_no_problems(self):
        assert make_question().validate() == []

    def test_chunk_text_is_required(self):
        """Requirement: the row must be reviewable without querying OpenSearch."""
        q = make_question(relevant_chunks=[RelevantChunk("2501.00001v1", 7, GRADE_ANSWER, chunk_text="")])
        assert any("no stored chunk_text" in p for p in q.validate())

    def test_answerable_row_needs_chunks(self):
        q = make_question(relevant_chunks=[], relevant_papers=[])
        assert any("no relevant_chunks" in p for p in q.validate())

    def test_unanswerable_row_must_have_no_chunks(self):
        q = make_question(answerable=False)
        assert any("unanswerable but lists" in p for p in q.validate())

    def test_relevant_papers_must_agree_with_chunks(self):
        q = make_question(relevant_papers=["2599.99999v1"])
        assert any("relevant_papers" in p for p in q.validate())

    def test_multi_hop_must_span_two_papers(self):
        q = make_question(question_type="multi_hop")
        assert any("must span >= 2 papers" in p for p in q.validate())

    def test_bad_grade_rejected(self):
        q = make_question(relevant_chunks=[RelevantChunk("2501.00001v1", 7, 5, chunk_text="t")])
        assert any("grade must be 1 or 2" in p for p in q.validate())

    def test_relevance_map_feeds_the_metrics(self):
        q = make_question(
            relevant_chunks=[
                RelevantChunk("2501.00001v1", 7, GRADE_ANSWER, chunk_text="a"),
                RelevantChunk("2501.00001v1", 8, GRADE_SUPPORTING, chunk_text="b"),
            ],
        )
        assert q.relevance_map() == {("2501.00001v1", 7): 2, ("2501.00001v1", 8): 1}

    def test_duplicate_qids_detected(self):
        assert any("duplicate qid" in p for p in validate_dataset([make_question(), make_question()]))

    def test_dangling_paraphrase_parent_detected(self):
        q = make_question("q0002", question_type="paraphrase", paraphrase_of="q9999")
        assert any("does not exist" in p for p in validate_dataset([q]))

    def test_is_accepted_only_for_accepted_and_edited(self):
        assert not make_question().is_accepted
        q = make_question()
        q.review["status"] = "accepted"
        assert q.is_accepted
        q.review["status"] = "rejected"
        assert not q.is_accepted


class TestPersistence:
    def test_round_trip_preserves_chunk_text(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        original = make_question(
            relevant_chunks=[RelevantChunk("2501.00001v1", 7, GRADE_ANSWER, chunk_text="the body", answer_snippet="snip")]
        )
        write_jsonl([original], path)

        loaded = read_jsonl(path)[0]
        assert loaded.qid == original.qid
        assert loaded.relevant_chunks[0].chunk_text == "the body"
        assert loaded.relevant_chunks[0].answer_snippet == "snip"

    def test_accepted_only_filters_rejected_rows(self, tmp_path):
        """Rejected rows stay in the file for provenance; the loader filters."""
        path = tmp_path / "golden.jsonl"
        keep, drop = make_question("q0001"), make_question("q0002")
        keep.review["status"] = "accepted"
        drop.review["status"] = "rejected"
        write_jsonl([keep, drop], path)

        assert [q.qid for q in read_jsonl(path, accepted_only=True)] == ["q0001"]
        assert len(read_jsonl(path)) == 2

    def test_missing_file_gives_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="generate"):
            read_jsonl(tmp_path / "nope.jsonl")

    def test_corrupt_line_names_the_line_number(self, tmp_path):
        """A malformed row must point at its line, not fail anonymously."""
        path = tmp_path / "bad.jsonl"
        write_jsonl([make_question("q0001")], path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("not json\n")

        with pytest.raises(ValueError, match=":2"):
            read_jsonl(path)


class TestJsonParsing:
    """Regression tests for the two bugs that produced zero negative controls.

    A full run asked for 20 unanswerable questions in one reply, the array was
    truncated by ``max_tokens``, and the fallback regex tried the object shape
    first — matching from the first object's brace to the last one's, which
    parses as "Extra data". Between them the whole batch was lost.
    """

    def test_plain_object(self):
        assert parse_json_response('{"question": "What?"}')["question"] == "What?"

    def test_fenced_object(self):
        assert parse_json_response('```json\n{"question": "What?"}\n```')["question"] == "What?"

    def test_array_of_objects_is_not_mangled(self):
        """The regression: object-first matching turned this into 'Extra data'."""
        raw = '[{"question": "A?"}, {"question": "B?"}]'
        parsed = parse_json_response(raw)
        assert isinstance(parsed, list)
        assert [entry["question"] for entry in parsed] == ["A?", "B?"]

    def test_array_with_commentary_around_it(self):
        raw = 'Here you go:\n[{"question": "A?"}, {"question": "B?"}]\nHope that helps.'
        assert len(parse_json_response(raw)) == 2

    def test_truncated_array_salvages_complete_objects(self):
        """max_tokens cut the reply mid-object; the complete ones still count."""
        raw = '[\n {"question": "A?", "kind": "out_of_scope"},\n {"question": "B?", "kind": "counterfactual"},\n {"question": "C?", "ki'
        parsed = parse_json_response(raw)
        assert [entry["question"] for entry in parsed] == ["A?", "B?"]

    def test_salvage_ignores_braces_inside_strings(self):
        raw = '[{"question": "What does {x} mean?"}, {"question": "B?"}]'
        assert [entry["question"] for entry in salvage_objects(raw)] == ["What does {x} mean?", "B?"]

    def test_salvage_handles_escaped_quotes(self):
        raw = '[{"question": "He said \\"hi\\" then?"}]'
        assert salvage_objects(raw)[0]["question"] == 'He said "hi" then?'

    def test_unparseable_output_raises(self):
        with pytest.raises(ValueError, match="No JSON found"):
            parse_json_response("I cannot help with that request.")


class TestHapaxIsIdentifierLike:
    """Regression tests for over-rejection on a small corpus.

    In a 1036-chunk corpus, ordinary words land in exactly one chunk by
    accident. Rejecting questions for using them cost four good questions on a
    full run.
    """

    @pytest.mark.parametrize("token", ["qwen1", "110b", "22b", "glm4", "llama3"])
    def test_identifiers_are_giveaways(self, token):
        assert is_identifier_like(token)

    @pytest.mark.parametrize("token", ["quantizing", "curates", "two-stage", "overlapped", "asr-based"])
    def test_ordinary_words_are_not_giveaways(self, token):
        """These were all wrongly rejected before the fix."""
        assert not is_identifier_like(token)

    def test_vocabulary_keeps_only_identifiers(self):
        chunks = ["the model quantizing weights", "another chunk entirely", "trained on qwen1 checkpoints"]
        hapax = build_hapax_vocabulary(chunks)
        assert "qwen1" in hapax
        assert "quantizing" not in hapax

    def test_ordinary_hapax_no_longer_rejects_a_question(self):
        chunks = ["a chunk that curates reading material", "unrelated text here"]
        hapax = build_hapax_vocabulary(chunks)
        verdict = check_leakage("Which agent curates reading material?", "unrelated source", hapax, 0.15)
        assert not verdict.leaks


class TestNearDuplicate:
    """Regression tests for near-duplicate negative controls.

    At temperature 0 an unchanged prompt returns the same questions, and
    exact-match dedup let surface variants through: a full run produced 20
    unanswerable questions that collapsed to 12 distinct probes, so refusal
    recall would have been measured on far fewer independent tests than the
    count implied.
    """

    def test_identical_is_duplicate(self):
        assert is_near_duplicate("What is the BLEU score?", ["What is the BLEU score?"])

    def test_swapped_number_is_still_duplicate(self):
        """The exact failure: only the parameter count changed."""
        existing = ["What was the BLEU score of the 70B-parameter variant on the WMT23 test set?"]
        assert is_near_duplicate("What was the BLEU score of the 1.2T-parameter variant on the WMT23 test set?", existing)

    def test_case_and_tense_variation_is_duplicate(self):
        existing = ["What was the F1 score of the vision transformer when applied to protein folding?"]
        assert is_near_duplicate("What is the F1 score of the Vision Transformer when applied to protein folding?", existing)

    def test_genuinely_different_question_passes(self):
        existing = ["What was the BLEU score of the 70B model on WMT23?"]
        assert not is_near_duplicate("What privacy budget was used in the federated medical imaging study?", existing)

    def test_same_metric_different_subject_passes(self):
        """Reusing 'F1 score' is fine if the subject genuinely differs."""
        existing = ["What was the F1 score of the vision transformer on protein folding prediction?"]
        assert not is_near_duplicate("What was the F1 score of the speech recognizer on noisy telephone audio?", existing)

    def test_empty_candidate_is_rejected(self):
        assert is_near_duplicate("", ["anything"])

    def test_first_question_always_passes(self):
        assert not is_near_duplicate("Any question at all here?", [])
