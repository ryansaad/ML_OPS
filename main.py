from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import os
import glob

app = FastAPI(title="Toxic Sentinel 2.0")

class Comment(BaseModel):
    text: str

model = None
CONFIDENCE_THRESHOLD = 0.90  # 👈 Only flag as Toxic if 90% sure

def find_model_path():
    files = glob.glob("**/MLmodel", recursive=True)
    if files:
        return os.path.dirname(files[0])
    return None

@app.on_event("startup")
async def load_model():
    global model
    model_path = find_model_path()
    if model_path:
        model = mlflow.sklearn.load_model(model_path)
        print(f"✅ Model loaded with 90% Confidence Threshold active.")
    else:
        print("❌ Model path not found!")

@app.post("/predict")
async def predict(comment: Comment):
    if model is None:
        return {"error": "Model not loaded yet. Please wait."}
    
    # Get probabilities for each class [Friendly, Toxic]
    # Example output: [[0.15, 0.85]]
    probabilities = model.predict_proba([comment.text])[0]
    toxic_prob = probabilities[1]
    
    # Apply the Confidence Filter
    if toxic_prob >= CONFIDENCE_THRESHOLD:
        prediction = "Toxic"
    else:
        prediction = "Friendly"
        
    return {
        "text": comment.text,
        "sentiment": prediction,
        "confidence": round(float(toxic_prob if prediction == "Toxic" else probabilities[0]), 4),
        "status": "Verified" if toxic_prob >= CONFIDENCE_THRESHOLD else "Filtered"
    }

@app.get("/")
async def root():
    return {"status": "Online", "version": "2.0-Refined"}
