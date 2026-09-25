from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_rejects_unsupported_format_before_touching_db():
    # The format check happens before any query, so no database is needed here.
    from app.api import documents

    app.dependency_overrides[documents.get_session] = lambda: None
    try:
        response = client.post(
            "/documents",
            files={"file": ("scan.pdf", b"%PDF-1.4", "application/pdf")},
            data={"owner": "test"},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 415
