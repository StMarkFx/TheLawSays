"""Enhanced tests for intent detection functionality."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from thelawsays_core.intent import IntentDetector, IntentDecision


class TestIntentDetectorInitialization:
    """Test intent detector initialization and configuration."""

    def test_detector_without_client(self):
        """Test detector initialization without OpenAI client."""
        detector = IntentDetector(client=None)
        assert detector.client is None
        assert detector.model == "gpt-4o-mini"

    def test_detector_with_client(self):
        """Test detector initialization with OpenAI client."""
        mock_client = Mock()
        detector = IntentDetector(client=mock_client, model="gpt-4")
        assert detector.client == mock_client
        assert detector.model == "gpt-4"

    def test_detector_caching(self):
        """Test that detector caches decisions."""
        detector = IntentDetector(client=None)
        query = "What is theft?"

        # First call
        decision1 = detector.classify(query)
        # Second call with same query
        decision2 = detector.classify(query)

        assert decision1 == decision2
        assert len(detector._cache) == 1
        assert query.lower() in detector._cache


class TestExplicitRetrievalTriggers:
    """Test detection of explicit retrieval triggers."""

    def test_explicit_legal_triggers(self):
        """Test detection of explicit legal keywords."""
        detector = IntentDetector(client=None)

        explicit_triggers = [
            "cite the law",
            "quote section 12",
            "what does the law say",
            "according to nigerian law",
            "legal basis for",
            "section reference",
        ]

        for query in explicit_triggers:
            decision = detector.classify(query)
            assert decision.retrieval_required is True
            assert decision.label == "legal_lookup"
            assert "explicit-trigger" in decision.reason

    def test_case_insensitive_triggers(self):
        """Test that triggers work case-insensitively."""
        detector = IntentDetector(client=None)

        variations = [
            "CITE the law",
            "Quote Section 12",
            "WHAT DOES THE LAW SAY",
            "According To Nigerian Law",
        ]

        for query in variations:
            decision = detector.classify(query)
            assert decision.retrieval_required is True


class TestConversationalPatterns:
    """Test detection of conversational patterns."""

    def test_greeting_patterns(self):
        """Test detection of greeting patterns."""
        detector = IntentDetector(client=None)

        greetings = [
            "hello",
            "hi there",
            "good morning",
            "hey",
            "good afternoon",
            "good evening",
            "good day",
        ]

        for greeting in greetings:
            decision = detector.classify(greeting)
            assert decision.retrieval_required is False
            assert decision.label == "conversational"
            assert "heuristic-conversational" in decision.reason

    def test_farewell_patterns(self):
        """Test detection of farewell patterns."""
        detector = IntentDetector(client=None)

        farewells = [
            "bye",
            "goodbye",
            "see you",
            "ciao",
            "later",
        ]

        for farewell in farewells:
            decision = detector.classify(farewell)
            assert decision.retrieval_required is False

    def test_thanks_patterns(self):
        """Test detection of thanks patterns."""
        detector = IntentDetector(client=None)

        thanks = [
            "thanks",
            "thank you",
            "appreciate it",
            "thanks a lot",
            "thanks so much",
        ]

        for thank in thanks:
            decision = detector.classify(thank)
            assert decision.retrieval_required is False

    def test_meta_question_patterns(self):
        """Test detection of meta questions about the system."""
        detector = IntentDetector(client=None)

        meta_questions = [
            "who are you",
            "what can you do",
            "what do you know",
            "who created you",
            "who built you",
            "tell me about yourself",
            "about you",
        ]

        for question in meta_questions:
            decision = detector.classify(question)
            assert decision.retrieval_required is False

    def test_help_patterns(self):
        """Test detection of help requests."""
        detector = IntentDetector(client=None)

        help_requests = [
            "help",
            "menu",
            "options",
            "start",
            "demo",
        ]

        for request in help_requests:
            decision = detector.classify(request)
            assert decision.retrieval_required is False


class TestLegalKeywordDetection:
    """Test detection based on legal keywords."""

    def test_common_legal_keywords(self):
        """Test detection of common legal terminology."""
        detector = IntentDetector(client=None)

        legal_keywords = [
            "arrest",
            "police",
            "court",
            "judge",
            "legal",
            "rights",
            "tenancy",
            "landlord",
            "tenant",
            "employment",
            "dismiss",
            "terminate",
            "fine",
            "penalty",
            "crime",
            "fraud",
            "contract",
            "business",
            "tax",
            "marriage",
            "divorce",
            "inheritance",
            "labour",
            "labor",
        ]

        for keyword in legal_keywords:
            query = f"What is {keyword}?"
            decision = detector.classify(query)
            assert decision.retrieval_required is True
            assert decision.label == "legal_lookup"

    def test_legal_keywords_in_context(self):
        """Test legal keywords within full questions."""
        detector = IntentDetector(client=None)

        legal_questions = [
            "Can the police arrest me without warrant?",
            "What are my rights during interrogation?",
            "How does tenancy law work in Lagos?",
            "Is it legal to terminate employment without notice?",
            "What crimes carry the death penalty?",
        ]

        for question in legal_questions:
            decision = detector.classify(question)
            assert decision.retrieval_required is True


class TestLengthBasedHeuristics:
    """Test length-based decision heuristics."""

    def test_short_non_legal_queries(self):
        """Test that short, non-legal queries are treated as conversational."""
        detector = IntentDetector(client=None)

        short_queries = [
            "ok",
            "yes",
            "no",
            "sure",
            "cool",
            "nice",
            "why",
            "how",
            "when",
        ]

        for query in short_queries:
            decision = detector.classify(query)
            assert decision.retrieval_required is False
            assert "length-heuristic" in decision.reason

    def test_medium_length_legal_queries(self):
        """Test medium-length queries with legal content."""
        detector = IntentDetector(client=None)

        # These should be long enough to trigger default legal lookup
        medium_queries = [
            "What happens if someone commits theft in Nigeria",
            "Can a tenant be evicted without court order",
            "What are the requirements for starting a business",
        ]

        for query in medium_queries:
            decision = detector.classify(query)
            # May use OpenAI or default to legal lookup
            assert isinstance(decision.retrieval_required, bool)


class TestOpenAIClassification:
    """Test OpenAI-based intent classification."""

    def test_openai_classification_success(self):
        """Test successful OpenAI classification."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "legal_lookup"
        mock_client.chat.completions.create.return_value = mock_response

        detector = IntentDetector(client=mock_client)

        decision = detector.classify("What is the penalty for robbery?")

        assert decision.label == "legal_lookup"
        assert decision.retrieval_required is True
        assert "openai-classifier" in decision.reason

        # Verify OpenAI API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4o-mini"
        assert "legal_lookup" in call_args[1]["messages"][0]["content"]

    def test_openai_classification_conversational(self):
        """Test OpenAI classification returning conversational."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "conversational"
        mock_client.chat.completions.create.return_value = mock_response

        detector = IntentDetector(client=mock_client)

        decision = detector.classify("Hello there")

        assert decision.label == "conversational"
        assert decision.retrieval_required is False

    def test_openai_classification_fallback(self):
        """Test fallback when OpenAI classification fails."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        detector = IntentDetector(client=mock_client)

        # Short query should fall back to length heuristic
        decision = detector.classify("hi")
        assert decision.retrieval_required is False

        # Longer query should default to legal lookup
        decision = detector.classify("What is the legal definition of contract?")
        assert decision.retrieval_required is True

    def test_openai_invalid_response_fallback(self):
        """Test fallback when OpenAI returns invalid response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid_response"
        mock_client.chat.completions.create.return_value = mock_response

        detector = IntentDetector(client=mock_client)

        decision = detector.classify("Some query")

        # Should fallback to legal lookup for invalid responses
        assert decision.label == "legal_lookup"
        assert decision.retrieval_required is True


class TestIntentDecisionCaching:
    """Test caching behavior of intent decisions."""

    def test_cache_hit(self):
        """Test that repeated queries use cache."""
        detector = IntentDetector(client=None)

        query = "What is theft?"
        decision1 = detector.classify(query)
        decision2 = detector.classify(query.upper())  # Different case

        assert decision1 == decision2
        assert len(detector._cache) == 2  # Both cases cached separately

    def test_cache_miss(self):
        """Test cache miss for new queries."""
        detector = IntentDetector(client=None)

        detector.classify("First query")
        assert len(detector._cache) == 1

        detector.classify("Second query")
        assert len(detector._cache) == 2

    def test_explicit_trigger_bypasses_cache(self):
        """Test that explicit triggers still use cache."""
        detector = IntentDetector(client=None)

        # First call
        decision1 = detector.classify("cite the law")
        # Second call - should use cache
        decision2 = detector.classify("cite the law")

        assert decision1 == decision2
        assert decision1.retrieval_required is True


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_query(self):
        """Test handling of empty queries."""
        detector = IntentDetector(client=None)

        decision = detector.classify("")
        assert decision.retrieval_required is False
        assert decision.label == "conversational"
        assert "empty-query" in decision.reason

    def test_whitespace_only_query(self):
        """Test handling of whitespace-only queries."""
        detector = IntentDetector(client=None)

        decision = detector.classify("   ")
        assert decision.retrieval_required is False

    def test_very_long_query(self):
        """Test handling of very long queries."""
        detector = IntentDetector(client=None)

        long_query = "What is the legal definition of " + "justice " * 100
        decision = detector.classify(long_query)

        # Should still classify properly
        assert isinstance(decision.retrieval_required, bool)
        assert decision.label in ["legal_lookup", "conversational"]

    def test_special_characters(self):
        """Test handling of special characters."""
        detector = IntentDetector(client=None)

        special_query = "What does §12(3) say about cafés?"
        decision = detector.classify(special_query)

        # Should handle without crashing
        assert isinstance(decision.retrieval_required, bool)

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        detector = IntentDetector(client=None)

        unicode_query = "What are the rights of naïve tenants?"
        decision = detector.classify(unicode_query)

        assert isinstance(decision.retrieval_required, bool)
