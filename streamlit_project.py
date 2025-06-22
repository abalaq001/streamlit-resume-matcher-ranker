import streamlit as st
import os
import re
import spacy
import joblib
import tempfile
import requests
from PyPDF2 import PdfReader
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from transformers import pipeline
from keybert import KeyBERT
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
import torch
import unicodedata

# ===========================
# ⚡️ 1️⃣ Load Dependencies
# ===========================
load_dotenv()
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
zero_shot_classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
nlp = spacy.load("en_core_web_sm")

# ===========================
# ⚡️ 2️⃣ Extract Text
# ===========================
def extract_text(file):
    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        return " ".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.read())
            tmp.flush()
            return docx2txt.process(tmp.name)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

# ===========================
# ⚡️ 3️⃣ Domain Prediction
# ===========================
CANDIDATE_DOMAINS = [
    "Data Science","Java Development","Web Development","Business Analysis","Marketing","Human Resources","Sales",
    "UI/UX Design","Project Management","Cybersecurity","Networking","Database Administration","Robotics","Embedded Systems","Machine Learning","Accounting",
    "Research","Mechanical Engineering","Electrical Engineering","Civil Engineering","Graphic Design","Content Writing",
    "Mobile App Development","Cloud Computing","Game Development","Artificial Intelligence","DevOps","Blockchain",
    "Quality Assurance","System Administration","Data Engineering","Full Stack Development","Software Testing","Digital Marketing",
    "SEO Specialist","Business Intelligence","E-commerce","Customer Support","Legal","Healthcare","Education","Consulting","Finance",
    "Logistics","Supply Chain Management","Public Relations","Non-profit","Telecommunications","Pharmaceuticals","Hospitality",
    "Retail","Insurance","Real Estate","Agriculture","Energy","Environmental Science","Aerospace","Fashion","Entertainment",
    "Sports Management","Travel and Tourism","Food and Beverage","Automotive","Construction","Mining","Petroleum Engineering",
    "Veterinary Medicine","Psychology","Social Work","Architecture","Data Analytics","Information Technology"
]

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def sanitize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text

def predict_domain_minilm(resume_text, candidate_labels=CANDIDATE_DOMAINS, top_k=1):
    resume_text = sanitize_text(resume_text)
    resume_embedding = embedding_model.encode(resume_text, convert_to_tensor=True)
    label_embeddings = embedding_model.encode(candidate_labels, convert_to_tensor=True)
    cos_scores = util.cos_sim(resume_embedding, label_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append({"domain": candidate_labels[idx], "similarity_score": float(score.item())})
    return results

# ===========================
# ⚡️ 4️⃣ Skill Extraction
# ===========================
def extract_skills(text):
    text = sanitize_text(text)
    doc = nlp(text)
    skills = set()
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 2:
            skills.add(token.text.lower())
    return sorted(skills)

# ===========================
# ⚡️ 5️⃣ Get Job Suggestions via JSearch API
# ===========================
def fetch_jobs(domain, limit=10):
    api_key = os.getenv("JSEARCH_API_KEY")
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {"X-RapidAPI-Host": "jsearch.p.rapidapi.com","X-RapidAPI-Key": api_key}
    params = {"query": f"{domain} jobs in India","page": "1","num_pages": "1"}
    response = requests.get(url, headers=headers, params=params)
    if response.ok:
        jobs_data = response.json().get("data", [])
        jobs = []
        for item in jobs_data:
            jobs.append({
                "title": item.get("job_title", "N/A"),
                "company": item.get("employer_name", "N/A"),
                "location": item.get("job_location", "N/A"),
                "url": item.get("job_apply_link", "#"),
                "description": item.get("job_description", "")
            })
        return jobs[:limit]
    else:
        st.error(f"Error fetching jobs: {response.text}")
        return []

# ===========================
# ⚡️ MAIN Streamlit App
# ===========================
st.title("🔍Resume Matcher and Ranker")
st.markdown("""
Upload your resume (PDF, DOCX, or TXT), and we'll:
- 👔 Predict your Domain.
- 🔥 Find and Rank Relevant Job Matches.
- ✅ Show Final Results with Apply Links.
""")

uploaded_file = st.file_uploader("📄 Upload your resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    resume_text = extract_text(uploaded_file)
    if not resume_text.strip():
        st.error("❌ Could not extract text. Try another file.")
        st.stop()

    predicted_domain_result = predict_domain_minilm(resume_text)
    predicted_domain = predicted_domain_result[0]["domain"] if predicted_domain_result else "Unknown"
    extracted_skills = extract_skills(resume_text)
    jobs = fetch_jobs(predicted_domain)

    indian_cities = ["bangalore","mumbai","chennai","delhi","kolkata","hyderabad","pune","gurgaon","noida","ahmedabad","kochi","lucknow"]
    jobs = [j for j in jobs if isinstance(j.get("location"), str) and ("india" in j.get("location").lower() or any(city in j.get("location").lower() for city in indian_cities))]
    if not jobs:
        st.warning(f"No India-specific jobs found for '{predicted_domain}', showing global results instead...")
        jobs = fetch_jobs(predicted_domain)

    if jobs:
        job_texts = [job["description"] + " " + job["title"] for job in jobs]
        all_texts = [resume_text] + job_texts
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        tfidf_matrix = tfidf.fit_transform(all_texts)
        resume_vector = tfidf_matrix[0]
        job_vectors = tfidf_matrix[1:]
        cosine_scores = cosine_similarity(resume_vector, job_vectors)[0]

        ranked_jobs = []
        expected_skills = set(extracted_skills)
        for idx, job in enumerate(jobs):
            cos_score = float(cosine_scores[idx])
            job_skills = extract_skills(job["description"])
            skill_overlap = len(expected_skills.intersection(job_skills)) / len(expected_skills) if expected_skills else 0
            final_score = 0.7 * cos_score + 0.3 * skill_overlap
            ranked_jobs.append({
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "url": job["url"],
                "description": job["description"],
                "cosine_score": cos_score,
                "skill_score": skill_overlap,
                "final_score": final_score
            })

        ranked_jobs = sorted(ranked_jobs, key=lambda x: x["final_score"], reverse=True)

        # 📊 Plot cosine & skill scores
        job_titles = [job['title'] for job in ranked_jobs]
        cos_scores = [job['cosine_score'] for job in ranked_jobs]
        skill_scores = [job['skill_score'] for job in ranked_jobs]

        fig_cos = px.bar(
            x=job_titles,
            y=cos_scores,
            labels={'x': 'Job Title', 'y': 'Cosine Similarity'},
            title='Cosine Similarity Scores of Top Jobs',
            color=cos_scores,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cos, use_container_width=True)

        fig_skill = px.bar(
            x=job_titles,
            y=skill_scores,
            labels={'x': 'Job Title', 'y': 'Skill Match'},
            title='Skill Match Scores of Top Jobs',
            color=skill_scores,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_skill, use_container_width=True)

        st.success(f"✅ Final Predicted Domain: **{predicted_domain}**")
        st.markdown("### 🏆 Top Matching Jobs")
        for i, job in enumerate(ranked_jobs, 1):
            st.markdown(f"**{i}. {job['title']} at {job['company']}**")
            st.markdown(f"📍 Location: {job['location']}")
            st.markdown(
                f"💻 Cosine: `{job['cosine_score']:.4f}` | 🔧 Skill Match: `{job['skill_score']:.4f}` | 🏅 Final Score: `{job['final_score']:.4f}` | [💼 Apply Here]({job['url']})"
            )
            with st.expander("📄 Job Description"):
                st.write(job["description"])

        st.markdown("### 📊 How Scores Are Calculated")
        with st.expander("ℹ️ About Scoring"):
            st.markdown(
                """
                - **Cosine Score (70% weight)** – Measures textual similarity between your resume and job descriptions.
                - **Skill Score (30% weight)** – Measures overlap between your resume's skills and the job's requirements.
                - **Final Score** = 0.7 * Cosine + 0.3 * Skill Match.
                
                🔍 This combined score provides a balanced match so you can focus on the most relevant roles!
                """
            )
