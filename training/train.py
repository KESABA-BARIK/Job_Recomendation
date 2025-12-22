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
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

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

target_col = 'category'  # Change to 'job_title' if needed
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
num_samples = X.shape[0]
num_features = X.shape[1]

labels = ["Number of Samples", "Number of Skill Features"]
values = [num_samples, num_features]

plt.figure()
plt.bar(labels, values)
plt.title("Dataset Size vs Feature Dimensionality")
plt.xlabel("Metric")
plt.ylabel("Count")
plt.show()

skill_counts = X.sum(axis=0)

plt.figure()
plt.spy(X, markersize=1)
plt.title("Sparsity Pattern of Skill Matrix")
plt.show()


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

print("\n" + "=" * 50)
print("BUILDING JOB TITLE VECTORS")
print("=" * 50)

job_vec = {}              # title -> averaged vector
job_title_category = {}   # title -> category

for _, row in df.iterrows():
    title = row['job_title']      # ✅ correct column
    category = row['category']
    vec = mlb.transform([row['skills_list']])[0]

    if title not in job_vec:
        job_vec[title] = []
        job_title_category[title] = category

    job_vec[title].append(vec)

# Average vectors per job title
job_vec = {
    title: np.mean(vectors, axis=0)
    for title, vectors in job_vec.items()
}

print(f"Total job titles processed: {len(job_vec)}")

joblib.dump(job_vec, os.path.join(models_path, "job_title_vectors.pkl"))
joblib.dump(job_title_category, os.path.join(models_path, "job_title_category.pkl"))

print("✓ Job title vectors saved")

def recommend_job_titles(user_skills, top_n=5):
    valid_skills = [s.lower() for s in user_skills if s.lower() in mlb.classes_]

    if not valid_skills:
        return None, []

    user_vec = mlb.transform([valid_skills])

    # 1️⃣ Predict category
    predicted_category = model.predict(user_vec)[0]

    # 2️⃣ Rank job titles inside predicted category
    scores = []
    for title, vec in job_vec.items():
        if job_title_category[title] != predicted_category:
            continue

        sim = cosine_similarity(user_vec, vec.reshape(1, -1))[0][0]
        scores.append((title, round(float(sim), 3)))

    scores.sort(key=lambda x: x[1], reverse=True)

    return predicted_category, scores[:top_n]



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
print("JOB TITLE RECOMMENDATIONS")
print("=" * 50)

test_inputs = [
    ["python", "sql", "pandas"],
    ["java", "spring", "hibernate"],
    ["javascript", "react", "nodejs"],
    ["machine learning", "tensorflow", "python"]
]

for skills in test_cases:
    category, jobs = recommend_job_titles(skills, top_n=5)

    print("\nSkills:", skills)
    print("Predicted Category:", category)
    print("Top Job Titles:")
    for i, (title, score) in enumerate(jobs, 1):
        print(f"{i}. {title} (similarity: {score})")


print("\n" + "=" * 50)
print("✓ TRAINING COMPLETE!")
print("=" * 50)
print("\nNext steps:")
