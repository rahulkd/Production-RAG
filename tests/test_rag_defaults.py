"""Tests for the RAG routes' default top_k and answer sampling."""

import asyncio

import pytest
from pydantic import ValidationError
from src.config import BedrockSettings, Settings
from src.schemas.api.ask import AskRequest
from src.services.LLM.client import BedrockClient


def test_top_k_defaults_to_ten_and_is_still_capped_at_ten():
    assert AskRequest(query="q").top_k == 10
    assert AskRequest(query="q", top_k=3).top_k == 3
    with pytest.raises(ValidationError):
        AskRequest(query="q", top_k=11)


def test_rag_sampling_defaults_to_the_evaluated_values(monkeypatch):
    for key in ("BEDROCK__RAG_TEMPERATURE", "BEDROCK__RAG_TOP_P"):
        monkeypatch.delenv(key, raising=False)
    settings = BedrockSettings(_env_file=None)
    assert (settings.rag_temperature, settings.rag_top_p) == (0.0, 1.0)
    # The generic defaults for plain generate() calls are separate and untouched.
    assert (settings.temperature, settings.top_p) == (0.5, 0.9)


def test_rag_sampling_is_overridable_and_validated(monkeypatch):
    monkeypatch.setenv("BEDROCK__RAG_TEMPERATURE", "0.3")
    assert BedrockSettings(_env_file=None).rag_temperature == 0.3
    monkeypatch.setenv("BEDROCK__RAG_TEMPERATURE", "1.7")
    with pytest.raises(ValidationError):
        BedrockSettings(_env_file=None)


def test_both_rag_paths_pass_the_configured_sampling(monkeypatch):
    monkeypatch.setenv("BEDROCK__RAG_TEMPERATURE", "0.25")
    monkeypatch.setenv("BEDROCK__RAG_TOP_P", "0.8")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client = BedrockClient(Settings(bedrock=BedrockSettings(_env_file=None)))
    seen = []

    async def fake_generate(**kwargs):
        seen.append(kwargs)
        return {"response": '{"answer": "ok"}'}

    async def fake_stream(**kwargs):
        seen.append(kwargs)
        yield {"response": "ok", "done": True}

    monkeypatch.setattr(client, "generate", fake_generate)
    monkeypatch.setattr(client, "generate_stream", fake_stream)
    chunks = [{"arxiv_id": "2401.00001v1", "chunk_text": "text"}]

    asyncio.run(client.generate_rag_answer("q?", chunks))

    async def drain():
        async for _ in client.generate_rag_answer_stream("q?", chunks):
            pass

    asyncio.run(drain())
    assert len(seen) == 2
    assert all((call["temperature"], call["top_p"]) == (0.25, 0.8) for call in seen)
