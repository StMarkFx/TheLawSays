"""Tests for data structures and core utilities."""

from __future__ import annotations

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from thelawsays_core.data import Chunk, load_knowledge_base
from thelawsays_core.intent import IntentDecision


class TestChunk:
    """Test the Chunk data structure."""

    def test_chunk_creation(self):
        """Test creating a Chunk instance."""
        chunk = Chunk(
            text="Sample legal text",
            source="law.pdf",
            jurisdiction="Federal",
            meta={"section": "12", "page": "5"}
        )

        assert chunk.text == "Sample legal text"
        assert chunk.source == "law.pdf"
        assert chunk.jurisdiction == "Federal"
        assert chunk.meta == {"section": "12", "page": "5"}

    def test_chunk_equality(self):
        """Test chunk equality comparison."""
        chunk1 = Chunk("text", "source", "jurisdiction", {"key": "value"})
        chunk2 = Chunk("text", "source", "jurisdiction", {"key": "value"})
        chunk3 = Chunk("different", "source", "jurisdiction", {"key": "value"})

        assert chunk1 == chunk2
        assert chunk1 != chunk3

    def test_chunk_hash(self):
        """Test that chunks are hashable."""
        chunk = Chunk("text", "source", "jurisdiction", {})
        chunk_set = {chunk}
        assert len(chunk_set) == 1

        # Same content should hash the same
        chunk2 = Chunk("text", "source", "jurisdiction", {})
        chunk_set.add(chunk2)
        assert len(chunk_set) == 1

    def test_chunk_string_representation(self):
        """Test chunk string representation."""
        chunk = Chunk("Sample text", "doc.pdf", "Federal", {})
        str_repr = str(chunk)

        assert "Sample text" in str_repr
        assert "doc.pdf" in str_repr
        assert "Federal" in str_repr


class TestIntentDecision:
    """Test the IntentDecision data structure."""

    def test_intent_decision_creation(self):
        """Test creating IntentDecision instances."""
        decision = IntentDecision(
            label="legal_lookup",
            retrieval_required=True,
            reason="contains legal keywords"
        )

        assert decision.label == "legal_lookup"
        assert decision.retrieval_required is True
        assert decision.reason == "contains legal keywords"

    def test_conversational_decision(self):
        """Test conversational intent decisions."""
        decision = IntentDecision(
            label="conversational",
            retrieval_required=False,
            reason="greeting detected"
        )

        assert decision.label == "conversational"
        assert decision.retrieval_required is False

    def test_decision_equality(self):
        """Test decision equality."""
        d1 = IntentDecision("legal", True, "reason")
        d2 = IntentDecision("legal", True, "reason")
        d3 = IntentDecision("conversational", False, "reason")

        assert d1 == d2
        assert d1 != d3


class TestLoadKnowledgeBase:
    """Test knowledge base loading functionality."""

    @patch('thelawsays_core.data.Path')
    @patch('thelawsays_core.data.load_documents')
    @patch('thelawsays_core.data.load_bm25_index')
    @patch('thelawsays_core.data.load_faiss_index')
    @patch('thelawsays_core.data.SentenceTransformer')
    def test_load_knowledge_base_success(self, mock_transformer, mock_faiss, mock_bm25, mock_docs, mock_path):
        """Test successful knowledge base loading."""
        # Setup mocks
        mock_docs.return_value = [Chunk("text", "src", "jur", {})]
        mock_bm25.return_value = Mock()
        mock_faiss.return_value = Mock()
        mock_transformer.return_value = Mock()

        # Execute
        kb = load_knowledge_base()

        # Assert
        assert kb is not None
        mock_docs.assert_called_once()
        mock_bm25.assert_called_once()
        mock_faiss.assert_called_once()

    @patch('thelawsays_core.data.Path')
    @patch('thelawsays_core.data.load_documents')
    def test_load_knowledge_base_missing_files(self, mock_docs, mock_path):
        """Test handling of missing knowledge base files."""
        mock_path.return_value.exists.return_value = False
        mock_docs.side_effect = FileNotFoundError("documents.json not found")

        # Execute & Assert
        with pytest.raises(FileNotFoundError):
            load_knowledge_base()

    @patch('thelawsays_core.data.Path')
    @patch('thelawsays_core.data.load_documents')
    @patch('thelawsays_core.data.load_bm25_index')
    @patch('thelawsays_core.data.load_faiss_index')
    def test_load_knowledge_base_partial_loading(self, mock_faiss, mock_bm25, mock_docs, mock_path):
        """Test partial loading when some files are missing."""
        mock_docs.return_value = [Chunk("text", "src", "jur", {})]
        mock_bm25.side_effect = FileNotFoundError("bm25 missing")
        mock_faiss.return_value = Mock()

        # Should still attempt to load but fail on BM25
        with pytest.raises(FileNotFoundError):
            load_knowledge_base()


class TestChunkValidation:
    """Test chunk data validation."""

    def test_chunk_text_validation(self):
        """Test that chunks have valid text."""
        # Valid chunk
        chunk = Chunk("Valid text", "source", "jurisdiction", {})
        assert len(chunk.text.strip()) > 0

    def test_chunk_source_validation(self):
        """Test chunk source validation."""
        chunk = Chunk("text", "valid_source.pdf", "Federal", {})
        assert chunk.source.endswith('.pdf')

    def test_chunk_jurisdiction_validation(self):
        """Test jurisdiction field validation."""
        valid_jurisdictions = ["Federal", "Lagos"]

        for jur in valid_jurisdictions:
            chunk = Chunk("text", "source", jur, {})
            assert chunk.jurisdiction in valid_jurisdictions

    def test_chunk_meta_validation(self):
        """Test metadata validation."""
        meta = {"section": "12", "page": "5", "author": "Government"}
        chunk = Chunk("text", "source", "Federal", meta)

        assert isinstance(chunk.meta, dict)
        assert "section" in chunk.meta


class TestChunkSerialization:
    """Test chunk serialization/deserialization."""

    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = Chunk("text", "source.pdf", "Federal", {"key": "value"})

        # Chunks should be convertible to dict for JSON serialization
        chunk_dict = {
            "text": chunk.text,
            "source": chunk.source,
            "jurisdiction": chunk.jurisdiction,
            "meta": chunk.meta
        }

        assert chunk_dict["text"] == "text"
        assert chunk_dict["source"] == "source.pdf"
        assert chunk_dict["jurisdiction"] == "Federal"

    def test_chunk_json_serialization(self):
        """Test JSON serialization of chunks."""
        chunk = Chunk("text", "source", "Federal", {"key": "value"})

        # Should be JSON serializable
        chunk_data = {
            "text": chunk.text,
            "source": chunk.source,
            "jurisdiction": chunk.jurisdiction,
            "meta": chunk.meta
        }

        json_str = json.dumps(chunk_data)
        parsed = json.loads(json_str)

        assert parsed["text"] == "text"
        assert parsed["jurisdiction"] == "Federal"
