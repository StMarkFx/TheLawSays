"""Tests for configuration management."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from api.config import Settings, get_settings


class TestSettings:
    """Test settings configuration and validation."""

    def test_default_settings(self):
        """Test default settings values."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings()

            assert settings.environment == "development"
            assert settings.openai_model == "gpt-4o-mini"
            assert settings.retrieval_top_k == 5
            assert settings.retrieval_alpha == 0.65
            assert settings.enable_moderation is False
            assert settings.allow_origins == "http://localhost:3000"

    def test_environment_override(self):
        """Test environment variable overrides."""
        env_vars = {
            "ENVIRONMENT": "production",
            "OPENAI_API_KEY": "test-key-123",
            "OPENAI_MODEL": "gpt-4",
            "RETRIEVAL_TOP_K": "10",
            "RETRIEVAL_ALPHA": "0.8",
            "ENABLE_MODERATION": "true",
            "ALLOW_ORIGINS": "https://example.com,https://app.example.com"
        }

        with patch.dict("os.environ", env_vars):
            settings = Settings()

            assert settings.environment == "production"
            assert settings.openai_api_key == "test-key-123"
            assert settings.openai_model == "gpt-4"
            assert settings.retrieval_top_k == 10
            assert settings.retrieval_alpha == 0.8
            assert settings.enable_moderation is True
            assert settings.allow_origins == "https://example.com,https://app.example.com"

    def test_validation_constraints(self):
        """Test validation of settings constraints."""
        # Test retrieval_alpha bounds
        with patch.dict("os.environ", {"RETRIEVAL_ALPHA": "1.5"}):
            with pytest.raises(ValueError):
                Settings()

        with patch.dict("os.environ", {"RETRIEVAL_ALPHA": "-0.1"}):
            with pytest.raises(ValueError):
                Settings()

        # Test retrieval_top_k bounds
        with patch.dict("os.environ", {"RETRIEVAL_TOP_K": "0"}):
            with pytest.raises(ValueError):
                Settings()

        with patch.dict("os.environ", {"RETRIEVAL_TOP_K": "51"}):
            with pytest.raises(ValueError):
                Settings()

    def test_allow_origins_parsing(self):
        """Test parsing of comma-separated allow origins."""
        env_vars = {
            "ALLOW_ORIGINS": "http://localhost:3000, https://example.com , http://test.com"
        }

        with patch.dict("os.environ", env_vars):
            settings = Settings()
            assert settings.allow_origins == "http://localhost:3000, https://example.com , http://test.com"

    def test_missing_openai_key(self):
        """Test that missing OpenAI key raises error in production."""
        with patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=True):
            with pytest.raises(ValueError, match="openai_api_key"):
                Settings()


class TestGetSettings:
    """Test the get_settings function."""

    def test_get_settings_caching(self):
        """Test that get_settings caches results."""
        with patch.dict("os.environ", {"ENVIRONMENT": "test"}, clear=True):
            # Clear any existing cache
            get_settings.cache_clear()

            settings1 = get_settings()
            settings2 = get_settings()

            # Should be the same object due to caching
            assert settings1 is settings2

            # Verify it's actually cached by checking cache info
            assert get_settings.cache_info().hits >= 1

    def test_get_settings_environment_isolation(self):
        """Test that settings are properly isolated per environment."""
        get_settings.cache_clear()

        with patch.dict("os.environ", {"ENVIRONMENT": "dev", "RETRIEVAL_TOP_K": "3"}, clear=True):
            dev_settings = get_settings()
            assert dev_settings.environment == "dev"
            assert dev_settings.retrieval_top_k == 3

        # Change environment
        with patch.dict("os.environ", {"ENVIRONMENT": "prod", "RETRIEVAL_TOP_K": "7"}, clear=True):
            prod_settings = get_settings()
            assert prod_settings.environment == "prod"
            assert prod_settings.retrieval_top_k == 7

    def test_settings_immutability(self):
        """Test that settings objects are immutable after creation."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings()

            # Should not be able to modify settings
            with pytest.raises(Exception):  # Could be AttributeError or similar
                settings.environment = "modified"
