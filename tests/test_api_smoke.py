import pytest
import requests

API_BASE = "http://localhost:5000"  # Express server
ML_BASE = "http://localhost:8000"   # FastAPI server

def test_ml_health():
    """Test that the ML health endpoint returns JSON."""
    response = requests.get(f"{ML_BASE}/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert "models_loaded" in data

def test_ml_leaderboard():
    """Test that the leaderboard endpoint returns JSON."""
    response = requests.get(f"{ML_BASE}/leaderboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "player_name" in data[0]
        assert "rating" in data[0]

def test_express_proxy():
    """Test that Express correctly proxies ML requests."""
    response = requests.get(f"{API_BASE}/ml/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert "models_loaded" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 