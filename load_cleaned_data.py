
import pandas as pd
# Load the cleaned dataset
df = pd.read_csv(r"C:\Users\Samiya\OneDrive\vscode\resume_parser project\Cleaned_ResumeDataSet.csv")

# Show basic info
print("✅ Dataset loaded successfully!")






print("✅ Dataset loaded successfully!")
print(f"🔢 Number of samples: {len(df)}\n")

print("📋 Available columns:")
print(df.columns)
print(df[['Category', 'cleaned_resume']].head())

