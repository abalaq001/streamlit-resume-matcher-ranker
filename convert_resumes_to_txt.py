import os
import pdfplumber
from docx import Document

input_folder = "C:/Users/Samiya/OneDrive/vscode/resume_parser project/raw_resumes"

output_folder = "C:/Users/Samiya/OneDrive/vscode/resume_parser project/resumes_to_classify"

os.makedirs(output_folder, exist_ok=True)

def convert_pdf_to_txt(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def convert_docx_to_txt(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

for file_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file_name)
    name, ext = os.path.splitext(file_name)
    
    try:
        if ext.lower() == ".pdf":
            text = convert_pdf_to_txt(input_path)
        elif ext.lower() == ".docx":
            text = convert_docx_to_txt(input_path)
        else:
            print(f"❌ Skipping unsupported file: {file_name}")
            continue

        output_path = os.path.join(output_folder, f"{name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Converted: {file_name} → {name}.txt")
    except Exception as e:
        print(f"⚠️ Failed to convert {file_name}: {e}")
