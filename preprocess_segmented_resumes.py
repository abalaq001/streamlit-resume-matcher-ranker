import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (only if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Set paths
input_folder = r"C:\Users\Samiya\OneDrive\vscode\resume_parser project\resumes_to_classify"
output_folder = "preprocessed_resumes"
os.makedirs(output_folder, exist_ok=True)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

# Process all segmented resumes
for filename in os.listdir(input_folder):
    if filename.endswith('_segmented.txt'):
        with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as file:
            raw_text = file.read()
        
        cleaned = clean_text(raw_text)

        output_path = os.path.join(output_folder, filename.replace('_segmented.txt', '_preprocessed.txt'))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        print(f"✅ Preprocessed: {filename} → {os.path.basename(output_path)}")
