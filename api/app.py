from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# 🔥 Fix path for Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)

# Import prediction + loader
from model.predict_bert import predict, load_model


app = FastAPI(
    title="Mental Health AI API",
    description="Emotion Detection + Mental Health Scoring System",
    version="1.0"
)


# 🔥 LOAD MODEL AT STARTUP (VERY IMPORTANT)
@app.on_event("startup")
def startup_event():
    load_model()


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Mental Health AI API is running 🚀"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def get_prediction(data: TextInput):
    try:
        result = predict(data.text)
        return result
    except Exception as e:
        return {
            "error": str(e)
        }