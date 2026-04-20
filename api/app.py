from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# 🔥 Fix path for Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)

# Import prediction
from model.predict_bert import predict, load_model


app = FastAPI(
    title="Mental Health AI API",
    description="Emotion Detection + Mental Health Scoring System",
    version="1.0"
)


# 📥 Input schema
class TextInput(BaseModel):
    text: str


# 🚀 Load model at startup (optional but helps debug)
@app.on_event("startup")
def startup_event():
    try:
        print("Starting app and loading model...")
        load_model()
        print("Model loaded at startup!")
    except Exception as e:
        print("Startup model load failed:", str(e))


# 🏠 Root endpoint
@app.get("/")
def home():
    return {
        "message": "Mental Health AI API is running 🚀"
    }


# ❤️ Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# 🔥 MAIN PREDICTION ENDPOINT
@app.post("/predict")
def get_prediction(data: TextInput):
    result = predict(data.text)

    # ❗ If model returned error → raise proper HTTP error
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result