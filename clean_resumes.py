import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the resume dataset
df = pd.read_csv(r"C:\Users\Samiya\OneDrive\vscode\resume_parser project\UpdatedResumeDataSet.csv")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    # Lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Apply preprocessing to the resume column
df['cleaned_resume'] = df['Resume'].apply(clean_and_lemmatize)

# Save the cleaned data to a new CSV
df.to_csv("Cleaned_ResumeDataSet.csv", index=False)

print("✅ Cleaning and lemmatization done! Cleaned data saved to 'Cleaned_ResumeDataSet.csv'")
