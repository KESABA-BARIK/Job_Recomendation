import joblib
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(project_root, "models")

print(f"Loading models from: {models_path}")

try:
    model = joblib.load(os.path.join(models_path, "model.pkl"))
    mlb = joblib.load(os.path.join(models_path, "mlb.pkl"))
    job_vec = joblib.load(os.path.join(models_path, "job_title_vectors.pkl"))
    job_title_category = joblib.load(os.path.join(models_path, "job_title_category.pkl"))

    valid_skills = set(mlb.classes_)
    print(f"Models loaded successfully. Valid skills: {len(valid_skills)}")
    print(f"✓ Total job titles: {len(job_vec)}")
except Exception as e:
    print(f"ERROR loading models: {e}")
    raise


def predict(skills: list[str], top_n: int = 5):
    print(f"predict() called with skills: {skills}")
    skills = [s for s in skills if s in valid_skills]
    print(f"Valid skills after filtering: {skills}")

    if not skills:
        return "No valid skills provided"

    skills_encoded = mlb.transform([skills])
    prediction = model.predict(skills_encoded)[0]
    print(f"Prediction Category: {prediction}")

    scores = []
    for title, vec in job_vec.items():
        if job_title_category.get(title) != prediction:
            continue
        sim = cosine_similarity(skills_encoded, vec.reshape(1, -1))[0][0]
        scores.append({
            "job_title": title,
            "score": round(float(sim),3)
        })
        scores.sort(key=lambda x: x["score"], reverse=True)
    return {
        "Prediction": prediction,
        "recomended_jobs": scores[:top_n]
            }
