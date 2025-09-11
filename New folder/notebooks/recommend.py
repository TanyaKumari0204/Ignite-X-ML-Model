# notebooks/recommend.py
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import re

# Resolve path to data (assumes this file is in notebooks/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "internships.csv")

def _read_internships(path=CSV_PATH):
    # Try common separators and encodings
    for sep in [",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8", engine="python")
            # If read looks wrong (single column), continue trying
            if df.shape[1] > 1:
                break
        except Exception:
            continue
    else:
        # final attempt without specifying sep
        df = pd.read_csv(path, encoding="utf-8", engine="python")

    # Normalize column names and strip whitespace
    df.columns = df.columns.str.strip()

    # Ensure expected cols exist
    for col in ["title", "organization", "location", "requirements", "description", "mode", "duration_weeks", "stipend_per_month"]:
        if col not in df.columns:
            df[col] = ""

    # Normalize text fields
    text_cols = ["title", "organization", "location", "requirements", "description"]
    for c in text_cols:
        df[c] = df[c].astype(str).fillna("").str.strip()

    # Fix common mojibake (e.g. â€”)
    df["description"] = df["description"].str.replace("â€”", "—", regex=False)

    # Collect req_ columns if present
    req_cols = [c for c in df.columns if re.match(r"req_\d+", c)]
    # Create 'all_requirements' from req_ columns
    if req_cols:
        df["all_requirements"] = df[req_cols].fillna("").astype(str).agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()
    else:
        df["all_requirements"] = ""

    # Combined profile used for TF-IDF
    df["profile"] = (
        df["title"].fillna("") + " " +
        df["organization"].fillna("") + " " +
        df["requirements"].fillna("") + " " +
        df["all_requirements"].fillna("") + " " +
        df["description"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Coerce numeric columns (safe)
    if "stipend_per_month" in df.columns:
        df["stipend_per_month"] = (
            df["stipend_per_month"]
            .astype(str)
            .str.replace(",", "")
            .str.extract(r"(\d+)", expand=False)
            .fillna("")
        )
        df["stipend_per_month"] = pd.to_numeric(df["stipend_per_month"], errors="coerce")

    if "duration_weeks" in df.columns:
        df["duration_weeks"] = pd.to_numeric(df["duration_weeks"], errors="coerce")

    return df

# Load once at import time (fast for repeated calls)
_INTERN = _read_internships(CSV_PATH)

# Build TF-IDF matrix (use 'profile' field)
_vectorizer = TfidfVectorizer(stop_words="english")
# replace empty profile with a placeholder so vectorizer won't crash
_profiles = _INTERN["profile"].replace("", "empty")
_tfidf_matrix = _vectorizer.fit_transform(_profiles)


def recommend_for_candidate(candidate: dict, top_n: int = 5):
    """
    candidate: dict containing keys like:
        - education (str)
        - skills (str)            e.g. "Python, SQL, Machine Learning"
        - interests (str)
        - preferred_location (str)
        - mode (str)              e.g. "Onsite", "Remote"
        - min_stipend (int)       optional numeric filter
        - max_duration_weeks (int) optional numeric filter
    Returns: list of top_n recommendation dicts.
    """
    # Build candidate text profile
    candidate_profile = " ".join([
        str(candidate.get("education", "") or ""),
        str(candidate.get("skills", "") or ""),
        str(candidate.get("interests", "") or "")
    ]).strip()
    if candidate_profile == "":
        candidate_profile = "empty"

    # Compute cosine similarity
    try:
        cand_vec = _vectorizer.transform([candidate_profile])
        scores = cosine_similarity(cand_vec, _tfidf_matrix)[0]
    except Exception:
        # fallback to zeroes if transform fails
        scores = np.zeros(_INTERN.shape[0], dtype=float)

    results = _INTERN.copy()
    results["score"] = scores.astype(float)

    # Optional filtering
    # mode filter (if provided)
    mode_pref = (candidate.get("mode") or "").strip().lower()
    if mode_pref:
        # keep those that contain the requested mode (case-insensitive)
        mask_mode = results["mode"].astype(str).str.lower().str.contains(mode_pref)
        if mask_mode.any():
            results = results[mask_mode]
        # if none match, we keep all (don't drop everything)

    # stipend filter (min)
    min_stipend = candidate.get("min_stipend")
    if min_stipend is not None and min_stipend != "":
        try:
            min_stipend = float(min_stipend)
            # filter only rows with stipend >= min_stipend (NaN treated as 0)
            stipend_vals = results["stipend_per_month"].fillna(0)
            mask_stipend = stipend_vals >= min_stipend
            if mask_stipend.any():
                results = results[mask_stipend]
        except Exception:
            pass

    # duration filter (max)
    max_duration = candidate.get("max_duration_weeks")
    if max_duration is not None and max_duration != "":
        try:
            max_duration = float(max_duration)
            duration_vals = results["duration_weeks"].fillna(np.inf)
            mask_dur = duration_vals <= max_duration
            if mask_dur.any():
                results = results[mask_dur]
        except Exception:
            pass

    # Boost score slightly for location match
    pref_loc = (candidate.get("preferred_location") or "").strip().lower()
    if pref_loc:
        results["score"] = results["score"] + results["location"].astype(str).str.lower().apply(lambda loc: 0.1 if pref_loc in loc else 0)

    # If after filters there are no rows, fall back to full dataset
    if results.shape[0] == 0:
        results = _INTERN.copy()
        results["score"] = scores.astype(float)

    # If top TF-IDF match is weak, do fuzzy fallback based on skills tokens
    if results["score"].max() < 0.20:
        # Parse user skills tokens (split by comma/semicolon)
        skills_text = (candidate.get("skills") or "").lower()
        tokens = [t.strip() for t in re.split(r"[;,]", skills_text) if t.strip()]
        fallback = []
        if tokens:
            for idx, row in results.iterrows():
                job_text = (" ".join([str(row["requirements"]), str(row.get("all_requirements", ""))])).lower()
                match_score = 0.0
                for t in tokens:
                    if not t:
                        continue
                    if t in job_text:
                        match_score += 1.0
                    else:
                        # similarity ratio against job_text (coarse)
                        ratio = difflib.SequenceMatcher(None, t, job_text).ratio()
                        if ratio >= 0.45:
                            match_score += 0.5
                if match_score > 0:
                    fallback.append((idx, match_score))
            if fallback:
                fallback = sorted(fallback, key=lambda x: x[1], reverse=True)[:top_n]
                recs = []
                for idx, ms in fallback:
                    row = _INTERN.loc[idx]
                    recs.append({
                        "title": row["title"],
                        "organization": row["organization"],
                        "location": row["location"],
                        "mode": row.get("mode", ""),
                        "duration_weeks": int(row["duration_weeks"]) if pd.notna(row["duration_weeks"]) else None,
                        "stipend_per_month": int(row["stipend_per_month"]) if pd.notna(row["stipend_per_month"]) else None,
                        "description": row.get("description", ""),
                        "requirements": row.get("requirements", ""),
                        "score": round(float(ms), 2)
                    })
                return recs

    # Normal TF-IDF ranking
    top = results.sort_values(by="score", ascending=False).head(top_n)
    recommendations = []
    for _, row in top.iterrows():
        recommendations.append({
            "title": row["title"],
            "organization": row["organization"],
            "location": row["location"],
            "mode": row.get("mode", ""),
            "duration_weeks": int(row["duration_weeks"]) if pd.notna(row["duration_weeks"]) else None,
            "stipend_per_month": int(row["stipend_per_month"]) if pd.notna(row["stipend_per_month"]) else None,
            "description": row.get("description", ""),
            "requirements": row.get("requirements", ""),
            "score": round(float(row["score"]), 2)
        })

    return recommendations