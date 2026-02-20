from app.main import health

def test_health():
    response = health()
    assert response["status"] == "ok"
