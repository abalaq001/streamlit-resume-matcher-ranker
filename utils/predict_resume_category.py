import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
with open("best_resume_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Label mapping (same as during training)
label_map = {
    0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain", 4: "Business Analyst",
    5: "Civil Engineer", 6: "Data Science", 7: "Database", 8: "DevOps Engineer", 9: "DotNet Developer",
    10: "ETL Developer", 11: "Electrical Engineering", 12: "HR", 13: "Hadoop", 14: "Health and fitness",
    15: "Java Developer", 16: "Mechanical Engineer", 17: "Network Security Engineer", 18: "Operations Manager",
    19: "PMO", 20: "Python Developer", 21: "SAP Developer", 22: "Sales", 23: "Testing", 24: "Web Designing"
}

# Preprocessing function (same as training)
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Predict function
def predict_resume_category(resume_text):
    cleaned_text = preprocess_text(resume_text)
    tfidf_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(tfidf_vector)[0]
    return label_map[prediction]

# === DEMO ===
if __name__ == "__main__":
    print("🔍 Enter or paste your resume text (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    resume_text = "\n".join(lines)

    print("\n🧠 Predicting category...")
    category = predict_resume_category(resume_text)
    print(f"✅ Predicted Category: **{category}**")
