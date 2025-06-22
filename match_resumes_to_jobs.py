import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 📂 Paths
resume_folder = r"C:\Users\Samiya\OneDrive\vscode\resume_parser project\resumes_to_classify\segmented_resumes"
job_file = r"C:\Users\Samiya\OneDrive\vscode\resume_parser project\job_descriptions.json"
output_file = "resume_job_matches.json"

# 📥 Load resumes
resumes = {}
for file_name in os.listdir(resume_folder):
    if file_name.endswith("_segmented.txt"):
        with open(os.path.join(resume_folder, file_name), "r", encoding="utf-8") as file:
            resumes[file_name] = file.read()

# 📥 Load job descriptions
with open(job_file, "r", encoding="utf-8") as f:
    jobs = json.load(f)

# ✅ Extract relevant job fields
jobs = [job for job in jobs if "job_description" in job and "job_title" in job]

job_texts = [
    job["job_description"] + " " + job["job_title"] + " " + job.get("employer_name", "")
    for job in jobs
]

job_titles = [job["job_title"] for job in jobs]

# Filter out empty resumes
resume_texts = [text for text in resumes.values() if text.strip() != ""]

# Filter out empty job descriptions
job_texts = [text for text in job_texts if text.strip() != ""]

# ⚠️ Check if we have data
if not resume_texts or not job_texts:
    print("🚨 No valid resumes or job descriptions found.")
    exit()

print(f"📝 Number of resumes: {len(resume_texts)}")
print(f"💼 Number of jobs: {len(job_texts)}")
print("🔍 First resume sample:", resume_texts[0][:200])
print("🔍 First job sample:", job_texts[0][:200])

# 🧠 TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
all_texts = resume_texts + job_texts
tfidf_matrix = vectorizer.fit_transform(all_texts)

resume_vectors = tfidf_matrix[:len(resume_texts)]
job_vectors = tfidf_matrix[len(resume_texts):]

# 🧮 Match resumes to jobs
matches = {}
resume_names = list(resumes.keys())
for i, resume_name in enumerate(resume_names):
    sim_scores = cosine_similarity(resume_vectors[i], job_vectors)[0]
    top_indices = np.argsort(sim_scores)[::-1][:5]
    top_matches = [{
        "job_title": jobs[idx]["job_title"],
        "company": jobs[idx].get("employer_name", ""),
        "location": jobs[idx].get("job_location", ""),
        "description_snippet": jobs[idx]["job_description"][:150] + "...",
        "score": round(float(sim_scores[idx]), 4)
    } for idx in top_indices]
    matches[resume_name] = top_matches

# 💾 Save results
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(matches, f, indent=4)

print(f"\n✅ Done! Top job matches saved to '{output_file}'")
