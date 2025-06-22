import streamlit as st
import joblib
import json
import tempfile
from PyPDF2 import PdfReader
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

# Load trained classifier and vectorizer
classifier = joblib.load("best_resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load domain skill mapping
from domain_skill_mapping import domain_skills

# Load jobs (India only)
with open("job_descriptions.json", "r", encoding="utf-8") as f:
    jobs = [j for j in json.load(f) if "India" in j.get("job_location", "")]

# Initialize KeyBERT
kw_model = KeyBERT("all-mpnet-base-v2")

# Define skill vocabulary for filtering
COMMON_SKILLS = set(skill.lower() for skills in domain_skills.values() for skill in skills)

# Extract text from uploaded resume
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
    return ""

# KeyBERT-based skill extraction
def extract_skills(text, top_n=15):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return sorted({kw[0].lower() for kw in keywords if kw[0].lower() in COMMON_SKILLS})

# Smart domain prediction based on skill overlap
def predict_domain(skills):
    overlap_counts = {
        domain: len(set(skills) & set(expected))
        for domain, expected in domain_skills.items()
    }
    if not overlap_counts or max(overlap_counts.values()) == 0:
        return None
    return max(overlap_counts, key=overlap_counts.get)

# App UI
st.title("🔍 Resume Matcher and Ranker")
st.markdown("Upload your resume to find the **best job matches** based on smart skill extraction, domain inference, and content similarity.")

uploaded_file = st.file_uploader("📄 Upload your resume", type=["pdf", "docx", "txt"])

if uploaded_file:
    resume_text = extract_text(uploaded_file)

    if not resume_text.strip():
        st.error("❌ Could not extract any text from the file.")
        st.stop()

    extracted_skills = extract_skills(resume_text)

    domain_from_skills = predict_domain(extracted_skills)
    domain_from_model = classifier.predict(vectorizer.transform([resume_text]))[0]
    final_domain = domain_from_skills if domain_from_skills else domain_from_model

    if domain_from_skills and domain_from_skills != domain_from_model:
        st.warning(f"🔁 Domain prediction overridden: `{domain_from_model}` ➝ `{domain_from_skills}`")

    st.success(f"✅ Final Predicted Domain: {final_domain}")

    expected = set(domain_skills.get(final_domain, []))
    matched = expected & set(extracted_skills)
    skill_score = len(matched) / len(expected) if expected else 0.0

    # Job ranking
    job_texts = [job["job_description"] + " " + job["job_title"] for job in jobs]
    tfidf_matrix = vectorizer.fit_transform([resume_text] + job_texts)
    resume_vec = tfidf_matrix[0]
    job_vecs = tfidf_matrix[1:]
    cosine_scores = cosine_similarity(resume_vec, job_vecs)[0]

    top_matches = []
    for i, job in enumerate(jobs):
        cos = float(cosine_scores[i])
        final_score = 0.7 * cos + 0.3 * skill_score
        if cos < 0.05:
            final_score *= 0.5
        top_matches.append({
            "title": job["job_title"],
            "company": job.get("employer_name", ""),
            "location": job.get("job_location", ""),
            "description": job["job_description"],
            "apply_link": job.get("apply_link", "#"),
            "cosine": round(cos, 4),
            "skills": round(skill_score, 4),
            "final": round(final_score, 4),
        })

    top_matches = sorted(top_matches, key=lambda x: x["final"], reverse=True)[:5]

    # Display matches
    st.markdown("### 🏆 Top 5 Matching Jobs")
    for i, match in enumerate(top_matches, 1):
        st.markdown(f"**{i}. {match['title']} at {match['company']}**")
        st.markdown(f"📍 Location: {match['location']}")
        st.markdown(f"🔗 [Apply Here]({match['apply_link']})")
        st.markdown(f"✅ Cosine Score: `{match['cosine']}` | ✅ Skill Score: `{match['skills']}` | 🏅 Final Score: `{match['final']}`")
        with st.expander("📄 Full Job Description"):
            st.write(match["description"])
