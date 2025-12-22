# app/model.py

import joblib
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(project_root, "models")

print(f"Loading models from: {models_path}")

try:
    model = joblib.load(os.path.join(models_path, "model.pkl"))
    mlb = joblib.load(os.path.join(models_path, "mlb.pkl"))
    valid_skills = mlb.classes_
    print(f"Models loaded successfully. Valid skills: {len(valid_skills)}")
except Exception as e:
    print(f"ERROR loading models: {e}")
    raise


def predict(skills: list[str]):
    print(f"predict() called with skills: {skills}")
    skills = [s for s in skills if s in valid_skills]
    print(f"Valid skills after filtering: {skills}")

    if not skills:
        return "No valid skills provided"

    skills_encoded = mlb.transform([skills])
    prediction = model.predict(skills_encoded)[0]
    print(f"Prediction: {prediction}")
    return prediction