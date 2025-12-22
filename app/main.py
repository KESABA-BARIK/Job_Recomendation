import traceback

from fastapi import FastAPI, HTTPException
from app.model import predict
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Job Recommender API running"}

class SkillsRequest(BaseModel):
    skills: list[str]
    top_n: int = 5

@app.post("/recommend")
def recommend_job(request: SkillsRequest):
    print("=== REQUEST RECEIVED ===")
    print(f"Skills: {request.skills}")
    try:
        result = predict(request.skills, request.top_n)
        print(f"Result: {result}")
        return {"recommended_job": result}
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "type": type(e).__name__
        }