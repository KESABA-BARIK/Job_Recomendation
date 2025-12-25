from fastapi.testclient import TestClient

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Job Recommender API running"}

def test_recommend_valid_skills():
    response = client.post("/recommend", json={"skills": ["python", "ml"]})
    assert response.status_code == 200
    assert "recommended_job" in response.json()
