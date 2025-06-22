import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned data
df = pd.read_csv(r"C:\Users\Samiya\OneDrive\vscode\resume_parser project\Cleaned_ResumeDataSet.csv")

# Initialize label encoder
label_encoder = LabelEncoder()

# Fit and transform the 'Category' column
df['encoded_category'] = label_encoder.fit_transform(df['Category'])

# Save label mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("✅ Label encoding complete!\n🔢 Label Mapping:")
for label, code in label_mapping.items():
    print(f"  {label}: {code}")

# Save updated dataframe
df.to_csv("Encoded_ResumeDataSet.csv", index=False)
print("\n📁 Saved to 'Encoded_ResumeDataSet.csv'")
