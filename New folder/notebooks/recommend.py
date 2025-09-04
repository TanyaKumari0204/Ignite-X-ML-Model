import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ...existing code...
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "data", "internships.csv")
internships = pd.read_csv(csv_path, sep="\t")
# Load internships with proper separator and handle missing values
#internships = pd.read_csv("data/internships.csv", sep="\t")
for col in ["title", "organization", "location", "requirements"]:
    if col not in internships.columns:
        internships[col] = ""
    else:
        internships[col] = internships[col].fillna("")

# Create a combined profile column
internships["profile"] = internships["title"] + " " + internships["organization"] + " " + internships["requirements"]

# TF-IDF vectorizer (ignore empty strings)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(internships["profile"].replace("", "empty"))

def recommend_for_candidate(candidate: dict, top_n=5):
    """
    candidate: dict with keys 'education', 'skills', 'interests', 'preferred_location'
    top_n: number of top recommendations
    """
    # Build candidate profile
    candidate_profile = " ".join([
        candidate.get("education", ""),
        candidate.get("skills", ""),
        candidate.get("interests", "")
    ])
    
    if candidate_profile.strip() == "":
        candidate_profile = "empty"
    
    # Compute similarity
    candidate_vector = vectorizer.transform([candidate_profile])
    scores = cosine_similarity(candidate_vector, tfidf_matrix)[0]
    
    results = internships.copy()
    results["score"] = scores
    
    # Boost if preferred location matches
    pref_loc = candidate.get("preferred_location", "").lower()
    results["score"] += results["location"].apply(lambda loc: 0.1 if pref_loc in loc.lower() else 0)
    
    # Top N recommendations
    top_internships = results.sort_values(by="score", ascending=False).head(top_n)
    
    recommendations = []
    for _, row in top_internships.iterrows():
        recommendations.append({
            "title": row["title"],
            "organization": row["organization"],
            "location": row["location"],
            "score": round(row["score"], 2)
        })
    
    return recommendations
