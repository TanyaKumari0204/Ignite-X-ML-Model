# app.py (Streamlit UI)
import streamlit as st
from notebooks.recommend import recommend_for_candidate

st.set_page_config(page_title="Internship Recommendation System", layout="wide")
st.title("💼 Internship Recommendation System (Streamlit)")

with st.form("candidate_form"):
    name = st.text_input("Name")
    education = st.text_input("Education")
    skills = st.text_input("Skills (comma or semicolon separated)")
    interests = st.text_input("Interests")
    preferred_location = st.text_input("Preferred Location")
    mode = st.selectbox("Preferred Mode", options=["", "Onsite", "Remote", "Hybrid"])
    min_stipend = st.number_input("Minimum stipend per month (leave 0 for no filter)", min_value=0, value=0)
    max_duration = st.number_input("Maximum duration (weeks) (leave 0 for no filter)", min_value=0, value=0)
    top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    candidate = {
        "education": education,
        "skills": skills,
        "interests": interests,
        "preferred_location": preferred_location,
        "mode": mode,
        "min_stipend": int(min_stipend) if min_stipend and min_stipend > 0 else None,
        "max_duration_weeks": int(max_duration) if max_duration and max_duration > 0 else None
    }

    try:
        recs = recommend_for_candidate(candidate, top_n=top_n)
        if recs:
            st.subheader(f"Top {len(recs)} recommendations for {name or 'Candidate'}")
            for i, r in enumerate(recs, 1):
                st.markdown(f"### {i}. {r['title']} — {r['organization']}")
                st.write(f"📍 **Location:** {r.get('location', '')}  |  **Mode:** {r.get('mode', '')}")
                if r.get("duration_weeks") is not None:
                    st.write(f"⏳ **Duration (weeks):** {r.get('duration_weeks')}")
                if r.get("stipend_per_month") is not None:
                    st.write(f"💰 **Stipend (per month):** ₹{r.get('stipend_per_month')}")
                if r.get("requirements"):
                    st.write(f"🔧 **Requirements:** {r.get('requirements')}")
                if r.get("description"):
                    st.write(f"📝 {r.get('description')}")
                st.write(f"⭐ **Match score:** {r.get('score')}")
                st.write("---")
        else:
            st.warning("No recommendations found. Try different skills or remove strict filters.")
    except Exception as e:
        st.error(f"Error while getting recommendations: {e}")
