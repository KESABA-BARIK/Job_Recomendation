import os
import joblib
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(project_root, "models")
os.makedirs(models_path, exist_ok=True)

# Load your dataset
print("Loading dataset...")
df = pd.read_csv("all_job_post.csv")  # UPDATE THIS PATH
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(3))

# Data Cleaning
print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

# Remove rows with missing skills or job info
df = df.dropna(subset=['job_skill_set', 'category'])  # Use 'job_title' if predicting titles
print(f"Rows after removing nulls: {len(df)}")


# Parse skills from string representation of list
def parse_skills(skill_str):
    """Convert skill string to list of cleaned skills"""
    if pd.isna(skill_str):
        return []

    # If it's already a list, return as is
    if isinstance(skill_str, list):
        return [s.strip().lower() for s in skill_str]

    # Parse string representation of list like "['python','java']"
    try:
        skills = ast.literal_eval(str(skill_str))
        if isinstance(skills, list):
            return [s.strip().lower() for s in skills if isinstance(s, str) and s.strip()]
    except:
        # Fallback: try comma separation
        skills = str(skill_str).replace('[', '').replace(']', '').replace("'", "").replace('"', '').split(',')
        return [s.strip().lower() for s in skills if s.strip()]

    return []


print("Parsing skills...")
df['skills_list'] = df['job_skill_set'].apply(parse_skills)

# Show sample parsed skills
print("\nSample parsed skills:")
for i in range(min(3, len(df))):
    print(f"{df['job_skill_set'].iloc[i]} -> {df['skills_list'].iloc[i]}")

# Remove rows with empty skills
df = df[df['skills_list'].apply(len) > 0]
print(f"\nRows after removing empty skills: {len(df)}")

# Exploratory Analysis
print("\n" + "=" * 50)
print("DATA EXPLORATION")
print("=" * 50)

target_col = 'job_title'  # Change to 'job_title' if needed
print(f"\nTarget column: {target_col}")
print(f"\nJob distribution:")
job_counts = df[target_col].value_counts()
print(job_counts)

# Check if any class has too few examples
min_samples = job_counts.min()
print(f"\nMinimum samples per class: {min_samples}")
if min_samples < 5:
    print("⚠️  WARNING: Some classes have <5 examples. Consider removing them.")
    # Optionally remove rare classes
    # rare_jobs = job_counts[job_counts < 5].index
    # df = df[~df[target_col].isin(rare_jobs)]
    # print(f"Rows after removing rare classes: {len(df)}")

# Get all unique skills
all_skills = set()
for skills in df['skills_list']:
    all_skills.update(skills)
print(f"\nTotal unique skills: {len(all_skills)}")
print(f"\nSample skills (first 30):")
print(sorted(list(all_skills))[:30])

# Feature Engineering
print("\n" + "=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['skills_list'])
y = df[target_col]

print(f"Feature matrix shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of skill features: {X.shape[1]}")

# Train-Test Split
print("\n" + "=" * 50)
print("TRAIN-TEST SPLIT")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Model Training
print("\n" + "=" * 50)
print("MODEL TRAINING")
print("=" * 50)

print("Training RandomForest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
model.fit(X_train, y_train)
print("✓ Training complete")

# Model Evaluation
print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"Training Accuracy: {train_acc:.2%}")

test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"Testing Accuracy: {test_acc:.2%}")

if test_acc < 0.5:
    print("⚠️  Low accuracy! Consider:")
    print("   - Getting more data")
    print("   - Feature engineering (use job_description)")
    print("   - Different model (try deep learning)")

print("\n" + "-" * 50)
print("Classification Report:")
print("-" * 50)
print(classification_report(y_test, test_pred))

# Feature Importance
print("\n" + "=" * 50)
print("TOP 20 IMPORTANT SKILLS")
print("=" * 50)

feature_importance = pd.DataFrame({
    'skill': mlb.classes_,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# Save Models
print("\n" + "=" * 50)
print("SAVING MODELS")
print("=" * 50)

joblib.dump(model, os.path.join(models_path, "model.pkl"))
joblib.dump(mlb, os.path.join(models_path, "mlb.pkl"))

print(f"✓ Model saved to: {os.path.join(models_path, 'model.pkl')}")
print(f"✓ MultiLabelBinarizer saved to: {os.path.join(models_path, 'mlb.pkl')}")
print(f"\nModel info:")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Testing accuracy: {test_acc:.2%}")
print(f"  - Number of skills: {len(mlb.classes_)}")
print(f"  - Number of job categories: {len(y.unique())}")

# Test with sample predictions
print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS")
print("=" * 50)

test_cases = [
    ["python", "sql", "pandas"],
    ["java", "spring", "mysql"],
    ["javascript", "react", "nodejs"],
    ["machine learning", "tensorflow", "python"]
]

for skills in test_cases:
    valid_skills = [s for s in skills if s in mlb.classes_]
    if valid_skills:
        encoded = mlb.transform([valid_skills])
        pred = model.predict(encoded)[0]
        proba = model.predict_proba(encoded)[0]
        confidence = max(proba)
        print(f"\nInput: {skills}")
        print(f"Valid skills used: {valid_skills}")
        print(f"Prediction: {pred} (confidence: {confidence:.1%})")
    else:
        print(f"\nInput: {skills}")
        print(f"⚠️  No valid skills found in model vocabulary")

print("\n" + "=" * 50)
print("✓ TRAINING COMPLETE!")
print("=" * 50)
print("\nNext steps:")
print("1. Check if accuracy is acceptable (>60% is decent for this task)")
print("2. Test the API with real queries")
print("3. If accuracy is low, consider using job_description text")