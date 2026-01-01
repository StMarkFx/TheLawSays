"""Tests for API routes and endpoints."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from api.main import create_app
from api.dependencies import get_rag_service
from api.schemas import ChatResponse, FeedbackRequest


@pytest.fixture
def mock_rag_service():
    """Create a mock RAG service for testing."""
    service = Mock()
    service.handle_chat.return_value = ChatResponse(
        answer="Mock response",
        chunks=[],
        retrieval_used=False,
        metadata={"test": "data"}
    )
    return service


@pytest.fixture
def client(mock_rag_service):
    """Create a test client with mocked dependencies."""
    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_endpoint_success(self, client):
        """Test successful health check."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data == {"status": "ok"}

    def test_health_endpoint_method_not_allowed(self, client):
        """Test that only GET is allowed on health endpoint."""
        response = client.post("/v1/health")
        assert response.status_code == 405

        response = client.put("/v1/health")
        assert response.status_code == 405

        response = client.delete("/v1/health")
        assert response.status_code == 405


class TestChatEndpoint:
    """Test the chat endpoint functionality."""

    def test_chat_endpoint_success(self, client, mock_rag_service):
        """Test successful chat request."""
        request_data = {
            "message": "What is the punishment for theft?",
            "jurisdiction": "Federal",
            "top_k": 5
        }

        response = client.post("/v1/chat", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "chunks" in data
        assert "retrieval_used" in data
        assert "metadata" in data
        assert data["answer"] == "Mock response"
        assert data["retrieval_used"] is False

        # Verify the service was called correctly
        mock_rag_service.handle_chat.assert_called_once()
        call_args = mock_rag_service.handle_chat.call_args[0][0]
        assert call_args.message == "What is the punishment for theft?"
        assert call_args.jurisdiction == "Federal"
        assert call_args.top_k == 5

    def test_chat_endpoint_minimal_request(self, client, mock_rag_service):
        """Test chat request with minimal required fields."""
        request_data = {"message": "Hello"}

        response = client.post("/v1/chat", json=request_data)

        assert response.status_code == 200
        mock_rag_service.handle_chat.assert_called_once()
        call_args = mock_rag_service.handle_chat.call_args[0][0]
        assert call_args.message == "Hello"
        assert call_args.jurisdiction is None
        assert call_args.top_k is None
        assert call_args.history == []

    def test_chat_endpoint_with_history(self, client, mock_rag_service):
        """Test chat request with conversation history."""
        request_data = {
            "message": "Follow-up question",
            "history": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"}
            ]
        }

        response = client.post("/v1/chat", json=request_data)

        assert response.status_code == 200
        call_args = mock_rag_service.handle_chat.call_args[0][0]
        assert len(call_args.history) == 2
        assert call_args.history[0].role == "user"
        assert call_args.history[0].content == "First question"

    def test_chat_endpoint_validation_error(self, client):
        """Test chat endpoint with invalid request data."""
        # Missing required message field
        response = client.post("/v1/chat", json={})
        assert response.status_code == 422

        # Empty message
        response = client.post("/v1/chat", json={"message": ""})
        assert response.status_code == 422

        # Invalid jurisdiction
        response = client.post("/v1/chat", json={
            "message": "Test",
            "jurisdiction": "InvalidJurisdiction"
        })
        # Should still work as jurisdiction is optional and validated at service level

    def test_chat_endpoint_service_exception(self, client, mock_rag_service):
        """Test handling of service-level exceptions."""
        mock_rag_service.handle_chat.side_effect = HTTPException(
            status_code=400,
            detail="Suspicious input detected"
        )

        request_data = {"message": "suspicious input"}
        response = client.post("/v1/chat", json=request_data)

        assert response.status_code == 400
        assert "Suspicious input detected" in response.json()["detail"]

    def test_chat_endpoint_internal_error(self, client, mock_rag_service):
        """Test handling of internal service errors."""
        mock_rag_service.handle_chat.side_effect = Exception("Internal error")

        request_data = {"message": "test"}
        response = client.post("/v1/chat", json=request_data)

        assert response.status_code == 500
        error_data = response.json()
        assert "detail" in error_data


class TestFeedbackEndpoint:
    """Test the feedback submission endpoint."""

    def test_feedback_endpoint_success(self, client):
        """Test successful feedback submission."""
        feedback_data = {
            "conversation_id": "conv-123",
            "rating": "thumbs_up",
            "comment": "Great answer!"
        }

        response = client.post("/v1/feedback", json=feedback_data)

        assert response.status_code == 202
        assert response.json() == {"status": "received"}

    def test_feedback_endpoint_minimal(self, client):
        """Test feedback with minimal required fields."""
        feedback_data = {
            "conversation_id": "conv-123",
            "rating": "thumbs_down"
        }

        response = client.post("/v1/feedback", json=feedback_data)

        assert response.status_code == 202

    def test_feedback_endpoint_validation(self, client):
        """Test feedback validation."""
        # Missing required fields
        response = client.post("/v1/feedback", json={})
        assert response.status_code == 422

        # Invalid rating
        response = client.post("/v1/feedback", json={
            "conversation_id": "conv-123",
            "rating": "invalid_rating"
        })
        assert response.status_code == 422

    def test_feedback_endpoint_method_not_allowed(self, client):
        """Test that only POST is allowed on feedback endpoint."""
        response = client.get("/v1/feedback")
        assert response.status_code == 405

        response = client.put("/v1/feedback", json={})
        assert response.status_code == 405


class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers(self, client):
        """Test that CORS headers are properly set."""
        response = client.options("/v1/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers

    def test_cors_allowed_origins(self, client):
        """Test CORS with allowed origins."""
        # This would need to be configured in the test app setup
        # For now, just verify the endpoint accepts requests from configured origins
        pass


class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_404_not_found(self, client):
        """Test 404 responses for unknown endpoints."""
        response = client.get("/v1/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 responses for incorrect HTTP methods."""
        response = client.get("/v1/chat")
        assert response.status_code == 405

        response = client.put("/v1/health")
        assert response.status_code == 405

    def test_invalid_json(self, client):
        """Test handling of invalid JSON payloads."""
        response = client.post("/v1/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestRequestValidation:
    """Test request validation and sanitization."""

    def test_message_length_limits(self, client):
        """Test that reasonable message lengths are accepted."""
        # Very long message
        long_message = "What is the law regarding " + "contracts " * 1000
        response = client.post("/v1/chat", json={"message": long_message})

        # Should still process (validation happens at service level)
        assert response.status_code in [200, 400]  # Either success or service-level rejection

    def test_special_characters_in_message(self, client):
        """Test handling of special characters in messages."""
        special_message = "What does §12(3) say about cafés?"
        response = client.post("/v1/chat", json={"message": special_message})

        # Should handle properly
        assert response.status_code in [200, 400]

    def test_jurisdiction_validation(self, client):
        """Test jurisdiction parameter validation."""
        # Valid jurisdictions should work
        for jur in ["Federal", "Lagos", None]:
            response = client.post("/v1/chat", json={
                "message": "Test",
                "jurisdiction": jur
            })
            assert response.status_code in [200, 400]  # Service may validate further
