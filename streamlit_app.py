import streamlit as st
import os
import tempfile
import joblib
import json
import numpy as np
import spacy
from PyPDF2 import PdfReader
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from domain_skill_mapping import domain_skills

# Load models
classifier = joblib.load(r"C:\\Users\\Samiya\\OneDrive\\vscode\\resume_parser project\\best_resume_classifier.pkl")
tfidf_vectorizer = joblib.load(r"C:\\Users\\Samiya\\OneDrive\\vscode\\resume_parser project\\tfidf_vectorizer.pkl")
nlp = spacy.load("en_core_web_sm")

# Load job descriptions
with open("job_descriptions.json", "r", encoding="utf-8") as f:
    jobs = json.load(f)
jobs = [job for job in jobs if "job_description" in job and "job_title" in job]

# Skill extractor (regex or SpaCy)
def extract_skills(text):
    doc = nlp(text)
    return set([token.text.lower() for token in doc if token.pos_ == "NOUN" and len(token.text) > 2])

# Read resume
@st.cache_data
def read_resume(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        text = " ".join([page.extract_text() or "" for page in pdf_reader.pages])
    elif uploaded_file.name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
    else:
        text = uploaded_file.read().decode("utf-8")
    return text

# Streamlit App
st.title("🔍 Resume Matcher and Ranker")
st.markdown("Upload your resume and get the top matching job profiles with smart ranking based on skills and job similarity.")

uploaded_file = st.file_uploader("📄 Upload your resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    resume_text = read_resume(uploaded_file)

    # Predict domain
    tfidf_resume_vector = tfidf_vectorizer.transform([resume_text])
    predicted_label = classifier.predict(tfidf_resume_vector)[0]
    predicted_domain = predicted_label if isinstance(predicted_label, str) else str(predicted_label)

    # Extract skills
    extracted_skills = extract_skills(resume_text)
    required_skills = set(domain_skills.get(predicted_domain, []))
    intersection = len(required_skills & extracted_skills)
    union = len(required_skills | extracted_skills)
    skill_score = intersection / union if union > 0 else 0.0

    # Job vectors
    job_texts = [job["job_description"] + " " + job["job_title"] + " " + job.get("employer_name", "") for job in jobs]
    tfidf_jobs = tfidf_vectorizer.transform(job_texts)

    # Cosine similarity
    cosine_scores = cosine_similarity(tfidf_resume_vector, tfidf_jobs)[0]

    # Combine scores
    combined_scores = []
    for i, job in enumerate(jobs):
        combined_score = 0.6 * cosine_scores[i] + 0.4 * skill_score
        combined_scores.append({
            "job_title": job["job_title"],
            "company": job.get("employer_name", "N/A"),
            "location": job.get("job_location", ""),
            "description": job["job_description"][:150] + "...",
            "cosine_score": round(float(cosine_scores[i]), 4),
            "skill_score": round(skill_score, 4),
            "combined_score": round(combined_score, 4)
        })

    combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)

    # Display results
    st.success(f"✅ Predicted Domain: {predicted_domain}")
    st.markdown(f"**Extracted Skills:** {', '.join(extracted_skills) if extracted_skills else 'None'}")

    st.markdown("---")
    st.subheader("🏆 Top 5 Matching Jobs")
    for i, match in enumerate(combined_scores[:5], 1):
        st.markdown(f"### {i}. {match['job_title']} at {match['company']}")
        st.markdown(f"📍 Location: {match['location']}")
        st.markdown(f"📝 Description: {match['description']}")
        st.markdown(f"✅ Cosine Score: `{match['cosine_score']}`")
        st.markdown(f"✅ Skill Match Score: `{match['skill_score']}`")
        st.markdown(f"🏅 Final Score: `{match['combined_score']}`")
        st.markdown("---")
