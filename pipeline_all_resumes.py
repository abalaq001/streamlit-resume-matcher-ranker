import os
import re
import spacy
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pdfplumber
import docx2txt

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

skill_keywords = [
    "python", "machine learning", "deep learning", "sql", "nlp", "data science",
    "c++", "tensorflow", "keras", "pandas", "matplotlib", "power bi", "excel",
    "html", "css", "javascript", "react", "vue", "flask", "aws", "docker",
    "figma", "adobe", "illustrator", "photoshop", "invision", "canva",
    "tableau", "pytorch", "jupyter", "colab", "opencv"
]

def extract_resume_text(file_path):
    if file_path.endswith('.pdf'):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        return text
    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path)
    else:
        return ""

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

def segment_resume_sections(text):
    section_keywords = {
        'summary': ['summary', 'objective', 'career objective'],
        'skills': ['skills', 'technical skills'],
        'experience': ['experience', 'work experience', 'professional experience'],
        'education': ['education', 'academic background', 'qualifications'],
        'projects': ['projects', 'academic projects'],
        'certifications': ['certifications'],
        'languages': ['languages', 'languages known'],
        'achievements': ['achievements', 'awards'],
    }

    lines = text.split('\n')
    section_data = {}
    current_section = None

    for line in lines:
        clean_line = line.strip().lower()
        found = False

        for key, keywords in section_keywords.items():
            for keyword in keywords:
                if keyword in clean_line and len(clean_line) < 50:
                    current_section = key
                    section_data[current_section] = []
                    found = True
                    break
            if found:
                break

        if current_section and not found and line.strip():
            section_data[current_section].append(line.strip())

    for key in section_data:
        section_data[key] = ' '.join(section_data[key])

    return section_data

def extract_phone(text):
    match = re.search(r"(\+91[-\s]?)?\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
    return match.group() if match else None

def extract_name(text):
    # Try first few lines
    lines = text.strip().split('\n')[:8]
    blacklist = {
        "python", "pandas", "tensorflow", "keras", "matplotlib", "jupyter", "notebook",
        "colab", "flask", "docker", "sql", "aws", "c++", "html", "css", "data", "science",
        "student", "resume", "email", "skills"
    }

    for line in lines:
        words = line.strip().split()
        # If the line has 2–3 words and they look like a human name
        if 1 < len(words) <= 3:
            if all(w[0].isupper() for w in words if w.isalpha()):
                if not any(w.lower() in blacklist for w in words):
                    return line.strip()

    # If not found, fallback to spaCy — but carefully
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if name.lower() not in blacklist and len(name.split()) <= 3:
                return name

    return "Unknown"




def extract_email(text):
    text = text.replace(" at ", "@").replace(" dot ", ".").replace(" [at] ", "@").replace(" [dot] ", ".")
    email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    matches = re.findall(email_regex, text)
    return matches[0] if matches else None

def extract_skills(text):
    text = text.lower()
    return list(set([skill for skill in skill_keywords if skill in text]))

def extract_education(text):
    edu_keywords = ["university", "college", "institute", "school", "b.tech", "btech", "m.tech", "msc", "bsc", "mba", "phd", "graduated"]
    lines = text.split('\n')
    education = []

    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in edu_keywords):
            if 5 < len(line.split()) < 20:
                education.append(line.strip())

    return education

def extract_entities(text):
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "education": extract_education(text)
    }

def run_full_pipeline(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf") or filename.endswith(".docx"):
            path = os.path.join(folder_path, filename)
            print(f"\nPROCESSING: {filename}")
            try:
                raw = extract_resume_text(path)
                clean = preprocess(raw)
                sections = segment_resume_sections(raw)
                entities = extract_entities(raw)

                print("\n ENTITIES:")
                for k, v in entities.items():
                    print(f"{k.capitalize()}: {v}")

                print("\n SECTIONS:")
                if not sections:
                    print("No clear sections found.")
                else:
                    for k, v in sections.items():
                        print(f"{k.capitalize()}: {v[:200]}...\n")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Set your folder path here
run_full_pipeline(r"C:\Users\Samiya\Downloads\resumes")
