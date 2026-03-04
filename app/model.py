from _datetime import datetime

import joblib
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

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
    normalized_skills = [s.lower().strip() for s in skills]
    skills = [s for s in normalized_skills if s in valid_skills]
    print(f"Valid skills after filtering: {skills}")

    if not skills:
        return {
            "error": "No valid skills provided",
            "skill_count" : len(valid_skills)
        }

    user_vec = mlb.transform([skills])
    probs = model.predict_proba(user_vec)[0]
    pred_index = np.argmax(probs)
    predicted_category = model.classes_[pred_index]
    confidence = round(float(probs[pred_index]),3)
    print(f"Prediction Category: {predicted_category}")

    scores = []
    for title, vec in job_vec.items():
        sim = cosine_similarity(user_vec, vec.reshape(1, -1))[0][0]

        # Boost if same predicted category
        category_boost = 0.1 if job_title_category.get(title) == predicted_category else 0

        final_score = (
                0.7 * sim +
                0.3 * confidence +
                category_boost
        )

        match_percentage = round(final_score * 100, 2)

        scores.append({
            "job_title": title,
            "match_percentage": match_percentage,
            "raw_similarity": round(float(sim), 3),
            "category": job_title_category.get(title)
        })
        scores.sort(key=lambda x: x["match_percentage"], reverse=True)

        for idx, item in enumerate(scores[:top_n]):
            item["rank"] = idx + 1
    return {
        "input_skills": skills,
        "Prediction": {
            "name": predicted_category,
            "confidence": confidence,
        },
        "recommended_jobs": scores[:top_n],
        "meta": {
            "total_jobs_scanned": len(job_vec),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model_version": "v2.0"
        }
        }

def extract_skills(text: str, skill_vocab: list):
    if not text:
        return []

    text = text.lower()

    extracted = []

    for skill in skill_vocab:
        # match full words only
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            extracted.append(skill)

    return list(set(extracted))