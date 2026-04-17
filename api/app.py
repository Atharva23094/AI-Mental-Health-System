from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# 🔥 Ensure correct path (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)

# Import prediction function
from model.predict_bert import predict


# 🚀 Initialize FastAPI app
app = FastAPI(
    title="Mental Health AI API",
    description="Emotion Detection + Mental Health Scoring System",
    version="1.0"
)


# 📥 Input schema
class TextInput(BaseModel):
    text: str


# 🏠 Root endpoint
@app.get("/")
def home():
    return {
        "message": "Mental Health AI API is running 🚀"
    }


# ❤️ Health check (VERY IMPORTANT for Render)
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# 🔥 MAIN PREDICTION ENDPOINT
@app.post("/predict")
def get_prediction(data: TextInput):
    try:
        result = predict(data.text)

        return {
            "input": data.text,
            "prediction": result
        }

    except Exception as e:
        return {
            "error": str(e)
        }