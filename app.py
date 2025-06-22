import streamlit as st
import os
import json
import spacy
import joblib
import tempfile
from PyPDF2 import PdfReader
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from domain_skill_mapping import domain_skills

# Load models
classifier = joblib.load("best_resume_classifier.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
nlp = spacy.load("en_core_web_sm")

# Load job data
with open("job_descriptions.json", "r", encoding="utf-8") as f:
    jobs = json.load(f)
    jobs = [j for j in jobs if j.get("job_title") and j.get("job_description")]

# Helper: Extract text from uploaded file
def extract_text(file):
    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.read())
            tmp.flush()
            return docx2txt.process(tmp.name)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""

# Helper: Extract skills

def extract_skills(text):
    doc = nlp(text)
    return set(token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2)

# Streamlit App
st.title("🔍 Resume Matcher and Ranker")
st.markdown("Upload your resume and get the top matching job profiles with smart ranking based on skills and job similarity.")

uploaded_file = st.file_uploader("📄 Upload your resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    resume_text = extract_text(uploaded_file)
    if not resume_text.strip():
        st.error("Could not extract text from the file. Please upload a valid PDF/DOCX/TXT file.")
        st.stop()

    # Predict category
    vector = tfidf_vectorizer.transform([resume_text])
    predicted_label = classifier.predict(vector)[0]

    # Skill extraction
    extracted_skills = extract_skills(resume_text)
    expected_skills = set(domain_skills.get(predicted_label, []))
    intersection = len(expected_skills & extracted_skills)
    union = len(expected_skills | extracted_skills)
    skill_score = intersection / union if union > 0 else 0.0

    # Match jobs
    job_texts = [
        job["job_description"] + " " + job["job_title"] + " " + job.get("employer_name", "")
        for job in jobs
    ]
    job_vectors = tfidf_vectorizer.transform(job_texts)
    resume_vector = tfidf_vectorizer.transform([resume_text])
    scores = cosine_similarity(resume_vector, job_vectors)[0]

    matches = []
    for i, job in enumerate(jobs):
        cosine_score = float(scores[i])
        combined = 0.6 * cosine_score + 0.4 * skill_score
        matches.append({
            "title": job["job_title"],
            "company": job.get("employer_name", ""),
            "location": job.get("job_location", ""),
            "description": job["job_description"],
            "cosine_score": round(cosine_score, 4),
            "skill_score": round(skill_score, 4),
            "final_score": round(combined, 4)
        })

    matches = sorted(matches, key=lambda x: x["final_score"], reverse=True)[:5]

    # Display Results
    st.success(f"✅ Predicted Domain: {predicted_label}")
    st.write("\n")
    st.markdown(f"**Extracted Skills:** {', '.join(extracted_skills) if extracted_skills else 'None'}")

    st.markdown("### 🏆 Top 5 Matching Jobs")
    for i, match in enumerate(matches, 1):
        st.markdown(f"**{i}. {match['title']} at {match['company']}**")
        st.markdown(f"📍 Location: {match['location']}")
        st.markdown(f"✅ Cosine Score: `{match['cosine_score']}`  |  ✅ Skill Match Score: `{match['skill_score']}`  |  🏅 Final Score: `{match['final_score']}`")
        with st.expander("📄 Full Job Description"):
            st.write(match['description'])
