import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load TF-IDF features and labels
with open("tfidf_features.pkl", "rb") as f:
    X = pickle.load(f)

df = pd.read_csv(r"C:\Users\Samiya\OneDrive\vscode\resume_parser project\Encoded_ResumeDataSet.csv")
y = df['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\n🔍 Training and evaluating models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📌 {name} Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-"*60)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Save best model
with open("best_resume_classifier.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\n✅ Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
print("📁 Saved to 'best_resume_classifier.pkl'")
