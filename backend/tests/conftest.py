from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def test_client():
    from app.main import create_app

    app = create_app()

    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    app.state.detector = mock_detector

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
