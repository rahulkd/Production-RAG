"""Tests for the Phase 5 response parsing (no HTTP)."""

import json

from src.evaluation.e2e.runner import paper_ids_from_sources, parse_stream


def frames(*payloads):
    return "".join(f"data: {json.dumps(p)}\n\n" for p in payloads)


def test_paper_ids_from_pdf_urls_are_normalized_and_deduplicated():
    sources = ["https://arxiv.org/pdf/2401.06831.pdf", "https://arxiv.org/pdf/2403.13269v3.pdf", "https://arxiv.org/pdf/2401.06831.pdf"]
    assert paper_ids_from_sources(sources) == ["2401.06831", "2403.13269"]


def test_parse_stream_reassembles_metadata_chunks_and_final_answer():
    body = frames(
        {"sources": ["https://arxiv.org/pdf/2401.06831.pdf"], "chunks_used": 3, "search_mode": "hybrid"},
        {"chunk": "SVM and "},
        {"chunk": "transfer learning."},
        {"answer": "SVM and transfer learning.", "done": True},
    )
    parsed = parse_stream(body)
    assert parsed["answer"] == "SVM and transfer learning."
    assert parsed["sources"] == ["https://arxiv.org/pdf/2401.06831.pdf"]
    assert (parsed["chunks_used"], parsed["search_mode"], parsed["error"]) == (3, "hybrid", None)


def test_parse_stream_handles_frames_split_across_reads():
    # aiter_text can cut mid-line; the runner joins the parts before parsing.
    body = frames({"chunk": "a"}, {"answer": "a", "done": True})
    assert parse_stream(body[:7] + body[7:])["answer"] == "a"


def test_parse_stream_no_chunks_reply():
    parsed = parse_stream(frames({"answer": "No relevant information found.", "sources": [], "done": True}))
    assert parsed["answer"] == "No relevant information found." and parsed["sources"] == []


def test_parse_stream_error_frame():
    parsed = parse_stream(frames({"error": "Search service unavailable"}))
    assert parsed["error"] == "Search service unavailable" and parsed["answer"] is None


def test_parse_stream_without_done_frame_keeps_partial_text():
    assert parse_stream(frames({"chunk": "partial "}, {"chunk": "answer"}))["answer"] == "partial answer"
