import streamlit as st
from notebooks.recommend import recommend_for_candidate

st.set_page_config(page_title="Internship Recommendation System", layout="wide")
st.title("💼 Internship Recommendation System")

# Candidate Inputs
with st.form("candidate_form"):
    name = st.text_input("Name")
    education = st.text_input("Education")
    skills = st.text_input("Skills (comma separated)")
    interests = st.text_input("Interests")
    preferred_location = st.text_input("Preferred Location")
    
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    # Build candidate dictionary
    candidate = {
        "education": education,
        "skills": skills,
        "interests": interests,
        "preferred_location": preferred_location
    }
    
    try:
        # Get recommendations
        recommendations = recommend_for_candidate(candidate, top_n=5)
        
        if recommendations:
            st.subheader(f"Top {len(recommendations)} Recommendations for {name}")
            for idx, rec in enumerate(recommendations, 1):
                st.markdown(
                    f"**{idx}. {rec['title']}** at **{rec['organization']}** "
                    f"({rec['location']}) | Score: {rec['score']}"
                )
        else:
            st.warning("No recommendations found. Please try with different skills/interests.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

