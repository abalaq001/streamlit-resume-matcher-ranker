from bert_skill_extractor import extract_skills_with_bert

sample_text = "Experienced in Python, TensorFlow, and AWS. Worked at Infosys on NLP applications."
skills = extract_skills_with_bert(sample_text)
print(skills)
