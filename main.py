import os
import glob
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Define a global variable for the model
model = None

def find_model_path():
    search_pattern = "/app/**/MLmodel"
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        files = glob.glob("**/MLmodel", recursive=True)
    return os.path.dirname(files[0]) if files else None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ON STARTUP
    global model
    print("📡 Lifespan Startup: Searching for model...")
    discovered_path = find_model_path()
    
    if discovered_path:
        print(f"✅ FOUND BRAIN AT: {discovered_path}")
        model = mlflow.sklearn.load_model(discovered_path)
        print("✅ Model loaded successfully. API is now ready for traffic.")
    else:
        print("❌ CRITICAL ERROR: No model found during lifespan startup!")
    
    yield
    # This code runs ON SHUTDOWN
    print("🛑 Lifespan Shutdown: Cleaning up...")

# Pass the lifespan handler to the FastAPI app
app = FastAPI(title="Toxic Sentinel", lifespan=lifespan)

class Comment(BaseModel):
    text: str

@app.post("/predict")
def predict(comment: Comment):
    if model is None:
        return {"error": "Model not loaded yet. Please wait."}
    prediction = model.predict([comment.text])
    sentiment = "Toxic" if prediction[0] == 1 else "Friendly"
    return {"sentiment": sentiment}

@app.get("/")
def home():
    return {"status": "Online"}
