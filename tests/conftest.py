import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="session")
def client():
    """Shared test client — loads model once for the whole session."""
    with TestClient(app) as c:
        yield c
