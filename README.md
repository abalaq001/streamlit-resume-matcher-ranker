# 🔍 Resume Matcher and Ranker

A **Streamlit-based web app** that matches resumes to job descriptions.  
This app:
- 🔍 Predicts the **domain** of a resume using **Sentence Transformers**.
- ⚡ Extracts **important skills** using **spaCy**.
- 📊 Fetches relevant **job listings** (focused on India) using the **JSearch API**.
- 🏆 Ranks jobs by **cosine similarity** and **skill overlap**.
- 📈 Displays a **bar chart** of top-ranked jobs.
- 📄 Provides direct **apply links** for each job.

---

## ✨ Features
✅ Domain Prediction via **MiniLM embeddings**  
✅ Job Fetching from **JSearch API**  
✅ Cosine Similarity Ranking of Jobs  
✅ Interactive **Streamlit UI** with file uploader and visualizations  
✅ Skill Extraction powered by **spaCy**  
✅ Filters jobs by **Indian locations**  

---

## 🧑‍💻 Tech Stack
- 🐍 **Python** 3.9+
- 📊 **Streamlit**
- 🧠 **Transformers** & **Sentence-Transformers**
- 🔧 **spaCy** NLP
- 📡 **JSearch API**
- 📂 **PyPDF2**, **docx2txt**
- 📐 **TfidfVectorizer** & **cosine_similarity**

---

## 🚀 Setup & Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/abalaq001/streamlit-resume-matcher-ranker.git
cd resume_parser project
