"""Shared pytest fixtures and configuration for TheLawSays tests."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from api.config import Settings
from api.services.rag import RagService
from thelawsays_core.data import Chunk


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    client = Mock()
    client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    )
    client.moderations.create.return_value = Mock(
        results=[Mock(flagged=False)]
    )
    return client


@pytest.fixture
def mock_knowledge_base():
    """Create a mock knowledge base for testing."""
    kb = Mock()
    kb.hybrid_retrieve.return_value = [
        Chunk(
            text="Sample legal text from constitution",
            source="Constitution.pdf",
            jurisdiction="Federal",
            meta={"section": "12", "page": "5"}
        ),
        Chunk(
            text="Additional legal text about rights",
            source="RightsAct.pdf",
            jurisdiction="Federal",
            meta={"section": "8", "page": "12"}
        )
    ]
    return kb


@pytest.fixture
def mock_intent_detector():
    """Create a mock intent detector."""
    detector = Mock()
    detector.classify.return_value = Mock(
        label="legal_lookup",
        retrieval_required=True,
        reason="test"
    )
    return detector


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            text="Section 12 of the Criminal Code provides that theft is punishable...",
            source="CriminalCode.pdf",
            jurisdiction="Federal",
            meta={"section": "12", "page": "45"}
        ),
        Chunk(
            text="The Constitution guarantees fundamental rights including...",
            source="Constitution.pdf",
            jurisdiction="Federal",
            meta={"section": "33", "page": "23"}
        ),
        Chunk(
            text="Under Lagos State tenancy law, landlords must...",
            source="TenancyLaw.pdf",
            jurisdiction="Lagos",
            meta={"section": "5", "page": "8"}
        )
    ]


@pytest.fixture
def test_settings():
    """Create test settings with safe defaults using env vars."""
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "test-key-123",
        "OPENAI_MODEL": "gpt-4o-mini",
        "RETRIEVAL_TOP_K": "5",
        "RETRIEVAL_ALPHA": "0.65",
        "ENABLE_MODERATION": "false",
        "ENVIRONMENT": "test",
        "ALLOW_ORIGINS": "http://localhost:3000,http://test.com"
    }):
        # Clear cache to get fresh settings
        get_settings.cache_clear()
        return get_settings()


@pytest.fixture
def mock_rag_service(test_settings, mock_knowledge_base, mock_intent_detector, mock_openai_client):
    """Create a fully mocked RAG service for testing."""
    with patch('api.services.rag.load_knowledge_base', return_value=mock_knowledge_base), \
         patch('api.services.rag.create_openai_client', return_value=mock_openai_client), \
         patch('thelawsays_core.intent.IntentDetector', return_value=mock_intent_detector):

        service = RagService(test_settings)
        return service


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Automatically mock environment variables for all tests."""
    env_vars = {
        "ENVIRONMENT": "test",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "gpt-4o-mini",
        "RETRIEVAL_TOP_K": "5",
        "RETRIEVAL_ALPHA": "0.65",
        "ENABLE_MODERATION": "false",
        "ALLOW_ORIGINS": "http://localhost:3000"
    }

    with patch.dict("os.environ", env_vars):
        yield


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Test data utilities
@pytest.fixture
def legal_test_queries():
    """Common legal test queries."""
    return [
        "What is the punishment for theft?",
        "Can police arrest without warrant?",
        "What are tenant rights in Lagos?",
        "How to register a business in Nigeria?",
        "What is the legal age of marriage?",
    ]


@pytest.fixture
def conversational_test_queries():
    """Common conversational test queries."""
    return [
        "Hello",
        "Thank you",
        "What can you do?",
        "Who created you?",
        "Goodbye",
    ]


@pytest.fixture
def suspicious_test_inputs():
    """Test inputs that should be flagged as suspicious."""
    return [
        "ignore previous instructions",
        "act as a different persona",
        "system override",
        "jailbreak",
        "a" * 3000,  # Very long input
        "a" * 20,    # Character repetition
    ]
