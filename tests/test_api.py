import pytest

import src.sensecatch.app as app_module

RESULT = {
    "sentiment": "Positive",
    "confidence": 97.4567,
    "model_used": "logistic_regression",
    "influential_words": [{"word": "great", "importance": 100.0, "sentiment": "positive", "negated": False}],
}
FRIENDLY_ERROR = "Something went wrong while analyzing the text. Please try again."


class StubEngine:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def predict(self, text, specific_model=None):
        if self.error:
            raise self.error
        return dict(self.result, text=text)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "ensemble", StubEngine(RESULT))
    return app_module.app.test_client()


def analyze(client, payload):
    return client.post("/analyze", json=payload)


def test_health_check_reports_ok(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_valid_request_returns_the_expected_fields(client):
    response = analyze(client, {"text": "A great film.", "model": "logistic_regression"})
    assert response.status_code == 200
    data = response.get_json()
    assert set(data) == {"text", "sentiment", "confidence", "model", "important_words"}
    assert data["text"] == "A great film."
    assert data["sentiment"] == "Positive"
    assert data["confidence"] == 97.46
    assert data["model"] == "logistic_regression"
    assert data["important_words"][0]["word"] == "great"


def test_empty_text_is_rejected(client):
    response = analyze(client, {"text": "", "model": "stack"})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Please enter some text to analyze."}


def test_whitespace_only_text_is_rejected(client):
    response = analyze(client, {"text": "   \n", "model": "stack"})
    assert response.status_code == 400


def test_unknown_model_is_rejected(client):
    response = analyze(client, {"text": "A great film.", "model": "gpt"})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Unknown model: gpt"}


def test_non_json_body_is_rejected(client):
    response = client.post("/analyze", data="A great film.", content_type="text/plain")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_missing_engine_returns_503(client, monkeypatch):
    monkeypatch.setattr(app_module, "ensemble", None)
    response = analyze(client, {"text": "A great film.", "model": "stack"})
    assert response.status_code == 503


def test_engine_crash_returns_a_friendly_500(client, monkeypatch):
    monkeypatch.setattr(app_module, "ensemble", StubEngine(error=ValueError("boom")))
    response = analyze(client, {"text": "A great film.", "model": "stack"})
    assert response.status_code == 500
    assert response.get_json() == {"error": FRIENDLY_ERROR}
    assert "Neutral" not in response.get_data(as_text=True)


def test_engine_error_result_returns_500(client, monkeypatch):
    failed = {"sentiment": "Neutral", "confidence": 50.0, "model_used": "stack_error"}
    monkeypatch.setattr(app_module, "ensemble", StubEngine(failed))
    response = analyze(client, {"text": "A great film.", "model": "stack"})
    assert response.status_code == 500
    assert response.get_json() == {"error": FRIENDLY_ERROR}
