# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from notebooks.recommend import recommend_for_candidate

app = FastAPI(title="Internship Recommendation API", version="1.0")


class CandidateRequest(BaseModel):
    education: Optional[str] = ""
    skills: Optional[str] = ""
    interests: Optional[str] = ""
    preferred_location: Optional[str] = ""
    mode: Optional[str] = ""               # e.g. "Onsite" or "Remote"
    min_stipend: Optional[float] = None
    max_duration_weeks: Optional[int] = None
    top_n: int = 5


@app.get("/")
def root():
    return {"message": "Internship Recommendation API is running"}


@app.post("/recommend")
def recommend(candidate: CandidateRequest):
    cand = candidate.dict()
    # rename keys to match internal names used by recommend_for_candidate
    # recommend_for_candidate expects a dict (we pass directly)
    try:
        recs = recommend_for_candidate(cand, top_n=cand.get("top_n", 5))
        return {"recommendations": recs}
    except Exception as e:
        return {"error": str(e)}
