import os
import json
import spacy
import numpy as np
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from domain_skill_mapping import domain_skills
from sklearn.preprocessing import LabelEncoder

# 📁 Paths
resume_folder = r"C:\\Users\\Samiya\\OneDrive\\vscode\\resume_parser project\\resumes_to_classify\\segmented_resumes"
job_file = r"C:\\Users\\Samiya\\OneDrive\\vscode\\resume_parser project\\job_descriptions.json"
classifier_file = r"C:\\Users\\Samiya\\OneDrive\\vscode\\resume_parser project\\best_resume_classifier.pkl"
tfidf_file = r"C:\\Users\\Samiya\\OneDrive\\vscode\\resume_parser project\\tfidf_vectorizer.pkl"
output_file = "ranked_resume_matches.json"

# ✅ Load pre-trained models
classifier = joblib.load(classifier_file)
tfidf_vectorizer = joblib.load(tfidf_file)

# ✅ Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# ✅ Load resumes
resumes = {}
for file_name in os.listdir(resume_folder):
    if file_name.endswith("_segmented.txt"):
        with open(os.path.join(resume_folder, file_name), "r", encoding="utf-8") as f:
            resumes[file_name] = f.read()

# ✅ Load job descriptions
with open(job_file, "r", encoding="utf-8") as f:
    jobs = json.load(f)
jobs = [job for job in jobs if "job_title" in job and "job_description" in job]

# ✅ Helper: Extract skills from text (simple matching against known skills)


# ✅ Replace the existing skill extractor with this one:

skill_keywords = {
    "python", "java", "c++", "html", "css", "javascript", "react", "angular", "node.js",
    "pandas", "numpy", "tensorflow", "scikit-learn", "django", "flask", "aws", "azure",
    "sql", "mysql", "git", "linux", "docker", "kubernetes", "rest api", "oop", "excel",
    "data analysis", "machine learning", "deep learning", "communication", "leadership"
}
skill_keywords.update({
    "statistics", "matplotlib", "tableau", "power bi", "user stories", "stakeholders",
    "responsive design", "wireframes", "adobe xd", "photoshop", "multithreading", "oop",
    "pytest", "sqlalchemy", "requirements gathering"
})


def extract_skills(text,required_skills):
    """
    Hybrid skill extractor using SpaCy NER + keyword matching
    """
    doc = nlp(text)
    
    # 1️⃣ Named entity skills (like Python, Java, etc.)
    ner_skills = set(
        [ent.text.strip().lower() for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT", "LANGUAGE"}]
    )

    # 2️⃣ Keyword-matched skills (fallback)
    keyword_matches = set()
    lowered_text = text.lower()
    for skill in skill_keywords:
        if skill in lowered_text:
            keyword_matches.add(skill)

    # 🔀 Combine both
    
    combined_skills = ner_skills.union(keyword_matches)

    irrelevant_phrases = {"b.tech", "bachelor", "university", "resume", "engineer", "student"}

    combined_skills = {
    skill for skill in combined_skills if skill not in irrelevant_phrases
}

    return combined_skills

# ✅ Predict domain and extract skills for each resume
ranked_matches = {}

# Flatten domain skills into a single set
all_known_skills = set()
for skills in domain_skills.values():
    all_known_skills.update(skills)

for resume_name, text in resumes.items():
    # TF-IDF vector for this resume
    tfidf_resume = tfidf_vectorizer.transform([text])

    # Predict domain
    text_vector = tfidf_vectorizer.transform([text])
    predicted_label = classifier.predict(text_vector)[0]
    predicted_domain = predicted_label if isinstance(predicted_label, str) else str(predicted_label)

    required_skills = set(domain_skills.get(predicted_domain, []))
    extracted_skills = extract_skills(text, required_skills)

 

    # Debug print
    print(f"\n🧪 Resume: {resume_name}")
    print(f"Predicted Domain: {predicted_domain}")
    print(f"Extracted Skills: {sorted(extracted_skills)}")
    print(f"Expected Skills: {sorted(required_skills)}")

    # Skill match score (Jaccard similarity)
    intersection = len(required_skills & extracted_skills)
    union = len(required_skills | extracted_skills)
    skill_score = intersection / union if union > 0 else 0.0

    # Calculate cosine similarity with jobs
    job_texts = [
        job.get("job_description", "").strip() + " " +
        job.get("job_title", "").strip() + " " +
        job.get("employer_name", "").strip()
        for job in jobs
        if job.get("job_description", "").strip() != ""
    ]

    if not job_texts:
        print("🚨 No job descriptions to process.")
        exit()

    tfidf_jobs = tfidf_vectorizer.transform(job_texts)

    # Calculate cosine similarity scores
    tfidf_scores = cosine_similarity(tfidf_resume, tfidf_jobs)[0]

    # Combine both scores
    combined_scores = [
        {
            "job_title": jobs[i]["job_title"],
            "company": jobs[i].get("employer_name", ""),
            "location": jobs[i].get("job_location", ""),
            "description_snippet": jobs[i]["job_description"][:150] + "...",
            "cosine_score": round(float(tfidf_scores[i]), 4),
            "skill_score": round(skill_score, 4),
            "combined_score": round(0.6 * tfidf_scores[i] + 0.4 * skill_score, 4)
        }
        for i in range(len(tfidf_scores))
    ]

    # Sort jobs by combined_score
    combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)

    ranked_matches[resume_name] = {
        "predicted_domain": predicted_domain,
        "extracted_skills": list(extracted_skills),
        "top_matches": combined_scores[:5]
    }

# ✅ Save results
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(ranked_matches, f, indent=4)

print(f"\n✅ Ranked resume-job matches saved to '{output_file}'")
