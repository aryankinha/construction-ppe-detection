from __future__ import annotations

import pytest

from app.core.config import Settings


def test_defaults():
    s = Settings()
    assert s.APP_ENV == "dev"
    assert s.DETECTION_CONFIDENCE == 0.5
    assert s.MODEL_PATH == "Model/ppe.pt"


def test_env_override(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("DETECTION_CONFIDENCE", "0.7")
    s = Settings()
    assert s.APP_ENV == "prod"
    assert s.DETECTION_CONFIDENCE == pytest.approx(0.7)
