"""Tests for prompt building utilities."""

from __future__ import annotations

import pytest

from thelawsays_core.prompts import (
    build_conversational_prompt,
    build_rag_prompt,
    build_no_results_prompt,
)
from thelawsays_core.data import Chunk


class TestConversationalPrompt:
    """Test building conversational prompts."""

    def test_simple_greeting(self):
        """Test prompt building for simple greetings."""
        prompt = build_conversational_prompt("Hello", [])
        assert "Hello" in prompt
        assert "legal research assistant" in prompt.lower()
        assert "Nigerian law" in prompt

    def test_with_history(self):
        """Test prompt building with conversation history."""
        history = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "What can you do?"}
        ]

        prompt = build_conversational_prompt("What can you do?", history)
        assert "What can you do?" in prompt
        assert "Hi there" in prompt
        assert "Hello! How can I help?" in prompt

    def test_empty_history(self):
        """Test prompt building with empty history."""
        prompt = build_conversational_prompt("Thanks", [])
        assert "Thanks" in prompt
        assert "Previous conversation:" not in prompt


class TestRagPrompt:
    """Test building RAG prompts with retrieved chunks."""

    def test_basic_rag_prompt(self):
        """Test basic RAG prompt structure."""
        chunks = [
            Chunk(
                text="Section 12 states that theft is punishable by imprisonment.",
                source="CriminalCode.pdf",
                jurisdiction="Federal",
                meta={"section": "12"}
            )
        ]

        prompt = build_rag_prompt("What is the punishment for theft?", chunks, "Federal", [])

        assert "What is the punishment for theft?" in prompt
        assert "Section 12 states that theft is punishable by imprisonment" in prompt
        assert "CriminalCode.pdf" in prompt
        assert "Federal" in prompt
        assert "jurisdiction" in prompt.lower()

    def test_multiple_chunks(self):
        """Test prompt with multiple retrieved chunks."""
        chunks = [
            Chunk(text="First legal text", source="doc1.pdf", jurisdiction="Federal", meta={}),
            Chunk(text="Second legal text", source="doc2.pdf", jurisdiction="Federal", meta={}),
        ]

        prompt = build_rag_prompt("Legal question", chunks, "Federal", [])

        assert "First legal text" in prompt
        assert "Second legal text" in prompt
        assert "doc1.pdf" in prompt
        assert "doc2.pdf" in prompt

    def test_with_history(self):
        """Test RAG prompt includes conversation history."""
        chunks = [Chunk(text="Legal content", source="law.pdf", jurisdiction="Federal", meta={})]
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        prompt = build_rag_prompt("Current question", chunks, "Federal", history)

        assert "Current question" in prompt
        assert "Previous question" in prompt
        assert "Previous answer" in prompt

    def test_jurisdiction_filtering(self):
        """Test that jurisdiction is properly included in prompt."""
        chunks = [Chunk(text="Legal text", source="law.pdf", jurisdiction="Lagos", meta={})]

        prompt = build_rag_prompt("Question", chunks, "Lagos", [])

        assert "Lagos" in prompt
        assert "jurisdiction" in prompt.lower()

    def test_empty_chunks(self):
        """Test behavior with empty chunks list."""
        prompt = build_rag_prompt("Question", [], "Federal", [])

        # Should still build a valid prompt even with no chunks
        assert "Question" in prompt
        assert isinstance(prompt, str)


class TestNoResultsPrompt:
    """Test building prompts when no results are found."""

    def test_basic_no_results(self):
        """Test basic no results prompt."""
        prompt = build_no_results_prompt("What is the meaning of life?", [])

        assert "What is the meaning of life?" in prompt
        assert "could not find" in prompt.lower() or "no relevant information" in prompt.lower()
        assert "Nigerian law" in prompt

    def test_no_results_with_history(self):
        """Test no results prompt with conversation history."""
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "I found some information about that"}
        ]

        prompt = build_no_results_prompt("Follow-up question", history)

        assert "Follow-up question" in prompt
        assert "First question" in prompt
        assert "I found some information about that" in prompt

    def test_edge_cases(self):
        """Test edge cases for no results prompt."""
        # Very short query
        prompt = build_no_results_prompt("Hi", [])
        assert "Hi" in prompt

        # Very long query
        long_query = "What is the legal definition of " + "justice " * 50
        prompt = build_no_results_prompt(long_query, [])
        assert long_query in prompt


class TestPromptStructure:
    """Test overall prompt structure and formatting."""

    def test_prompt_includes_system_instructions(self):
        """Test that all prompts include appropriate system instructions."""
        prompts = [
            build_conversational_prompt("Hello", []),
            build_rag_prompt("Legal question", [Chunk(text="content", source="src", jurisdiction="Fed", meta={})], "Federal", []),
            build_no_results_prompt("Question", [])
        ]

        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0
            # Should not contain excessive whitespace
            assert "\n\n\n" not in prompt

    def test_prompt_formatting(self):
        """Test that prompts are properly formatted."""
        chunks = [
            Chunk(text="Text with   multiple   spaces", source="test.pdf", jurisdiction="Federal", meta={})
        ]

        prompt = build_rag_prompt("Question", chunks, "Federal", [])

        # Should preserve important formatting but clean up excessive spaces
        assert "Text with   multiple   spaces" in prompt or "Text with multiple spaces" in prompt
        assert prompt.count("\n\n\n") == 0  # No excessive newlines

    def test_chunk_formatting_in_prompts(self):
        """Test how chunks are formatted within RAG prompts."""
        chunk = Chunk(
            text="Important legal text here",
            source="LawDocument.pdf",
            jurisdiction="Federal",
            meta={"section": "42", "page": "15"}
        )

        prompt = build_rag_prompt("What does the law say?", [chunk], "Federal", [])

        # Should include source information
        assert "LawDocument.pdf" in prompt
        assert "Federal" in prompt
        assert "Important legal text here" in prompt
