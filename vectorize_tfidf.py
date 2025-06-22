import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the cleaned and encoded dataset
df = pd.read_csv(r"C:\Users\Samiya\OneDrive\vscode\resume_parser project\Encoded_ResumeDataSet.csv")

# TF-IDF Vectorization
print("🔄 Performing TF-IDF vectorization on cleaned resumes...")
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['cleaned_resume'])

# Save the TF-IDF matrix and vectorizer
with open("tfidf_features.pkl", "wb") as f:
    pickle.dump(X, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ TF-IDF vectorization complete!")
print(f"🧮 Feature matrix shape: {X.shape}")
print("📁 TF-IDF data and vectorizer saved to 'tfidf_features.pkl' and 'tfidf_vectorizer.pkl'")
