from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import dependencies
from api.config import get_settings
from api.main import create_app
from api.routes import chat as chat_routes
from api.schemas import ChatResponse


class StubRagService:
    def handle_chat(self, payload):
        return ChatResponse(
            answer="stubbed response",
            chunks=[],
            retrieval_used=False,
            metadata={"intent_label": "conversational"},
        )


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("ENVIRONMENT", "test")
    get_settings.cache_clear()
    dependencies.get_rag_service.cache_clear()
    stub_service = StubRagService()
    monkeypatch.setattr(dependencies, "get_rag_service", lambda: stub_service)
    app = create_app()
    app.dependency_overrides[dependencies.get_rag_service] = lambda: stub_service
    app.dependency_overrides[chat_routes.get_rag_service] = lambda: stub_service
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_data(client: TestClient):
    response = client.post("/v1/chat", json={"message": "hello there"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "stubbed response"
    assert data["retrieval_used"] is False


def test_feedback_endpoint_accepts_payload(client: TestClient):
    payload = {
        "conversation_id": "abc123",
        "rating": "thumbs_up",
        "comment": "Great answer!",
    }
    response = client.post("/v1/feedback", json=payload)
    assert response.status_code == 202
    assert response.json() == {"status": "received"}
