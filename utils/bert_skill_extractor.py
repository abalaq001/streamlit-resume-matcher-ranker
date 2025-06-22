# bert_skill_extractor.py

import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ✅ Load pre-trained BERT model for NER (Force PyTorch backend)
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# ✅ Use PyTorch NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", framework="pt")

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^\w\s\-]", "", text)
    return text.strip()

def extract_skills_with_bert(text):
    clean = clean_text(text)
    ner_results = ner_pipeline(clean)

    extracted = [
        ent["word"].lower() for ent in ner_results
        if ent["entity_group"] in ["ORG", "MISC", "PER"] and len(ent["word"]) > 2
    ]

    return sorted(set(extracted))
def clean_extracted_skills(entities):
    # Remove stopwords, too short tokens, and irrelevant capital phrases
    bad_keywords = {"resume", "curriculum", "vitae", "name", "address", "email", "contact", "mobile"}
    cleaned = set()

    for e in entities:
        s = e.strip().lower()
        if len(s) > 2 and s not in bad_keywords and not any(char.isdigit() for char in s):
            # Remove entries with unusual characters
            if not re.search(r'[^a-z\s+/().-]', s):
                cleaned.add(s)

    return list(sorted(cleaned))

