from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from notebooks.recommend import recommend_for_candidate

app = FastAPI(title="Internship Recommendation API", version="1.0")

# Add CORS middleware after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class CandidateRequest(BaseModel):
    education: Optional[str] = ""
    skills: Optional[str] = ""
    interests: Optional[str] = ""
    preferred_location: Optional[str] = ""
    mode: Optional[str] = ""               # e.g. "Onsite" or "Remote"
    min_stipend: Optional[float] = None
    max_duration_weeks: Optional[int] = None
    # Additional filter parameters
    domain: Optional[str] = None
    education_level: Optional[str] = None
    max_stipend: Optional[float] = None
    top_n: int = 5

@app.get("/")
def root():
    return {"message": "Internship Recommendation API is running"}

@app.post("/recommend")
def recommend(candidate: CandidateRequest):
    cand = candidate.dict()
    try:
        recs = recommend_for_candidate(cand, top_n=cand.get("top_n", 5))
        return {"recommendations": recs}
    except Exception as e:
        return {"error": str(e)}