import traceback

from fastapi import FastAPI, HTTPException
from app.model import predict
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",  # Angular dev server
        #"https://your-frontend-domain.vercel.app"  # future deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/health")
def health():
    return {"status": "ok"}
