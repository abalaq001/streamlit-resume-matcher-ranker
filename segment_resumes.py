import os
import re

input_folder = r"C:\Users\Samiya\OneDrive\vscode\resume_parser project\resumes_to_classify"
output_folder = r"C:\Users\Samiya\OneDrive\vscode\resume_parser project\segmented_resumes"
os.makedirs(output_folder, exist_ok=True)

def segment_resume(text):
    sections = {
        "Education": [],
        "Experience": [],
        "Skills": [],
        "Projects": [],
        "Certifications": [],
        "Achievements": [],
        "Summary": []
    }

    current_section = None
    lines = text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower_line = stripped.lower()

        if re.search(r"\b(education|academic qualifications|educational background)\b", lower_line):
            current_section = "Education"
        elif re.search(r"\b(experience|work history|employment)\b", lower_line):
            current_section = "Experience"
        elif re.search(r"\b(skill|technical skills|languages)\b", lower_line):
            current_section = "Skills"
        elif re.search(r"\b(projects|project experience)\b", lower_line):
            current_section = "Projects"
        elif re.search(r"\b(certification|courses|training)\b", lower_line):
            current_section = "Certifications"
        elif re.search(r"\b(achievement|award|honors)\b", lower_line):
            current_section = "Achievements"
        elif re.search(r"\b(summary|objective|profile)\b", lower_line):
            current_section = "Summary"

        if current_section:
            sections[current_section].append(stripped)

    return sections

for file_name in os.listdir(input_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        segmented = segment_resume(content)

        # Save segmented sections to file
        output_path = os.path.join(output_folder, file_name.replace(".txt", "_segmented.txt"))
        with open(output_path, "w", encoding="utf-8") as out:
            for section, lines in segmented.items():
                if lines:
                    out.write(f"\n=== {section.upper()} ===\n")
                    out.write("\n".join(lines))
                    out.write("\n")

        print(f"✅ Segmented: {file_name} → {os.path.basename(output_path)}")
