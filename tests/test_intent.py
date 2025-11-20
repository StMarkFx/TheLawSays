from thelawsays_core.intent import IntentDetector
from thelawsays_core.openai_utils import create_openai_client


def test_intent_detector_conversational_without_api_key(monkeypatch):
    detector = IntentDetector(client=None)
    result = detector.classify("hi there")
    assert result.retrieval_required is False
    assert result.label == "conversational"


def test_intent_detector_detects_legal_phrase():
    detector = IntentDetector(client=None)
    result = detector.classify("Can the police arrest me for jaywalking?")
    assert result.retrieval_required is True
    assert result.label == "legal_lookup"
