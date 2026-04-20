from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import traceback

# Fix path for Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)

from model.predict_bert import predict


app = FastAPI(
    title="Mental Health AI API",
    description="Emotion Detection + Mental Health Scoring System",
    version="1.0"
)


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Mental Health AI API is running 🚀"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def get_prediction(data: TextInput):
    try:
        print("Incoming text:", data.text)

        result = predict(data.text)

        print("Prediction result:", result)

        return result

    except Exception as e:
        print("FULL ERROR:")
        traceback.print_exc()   # 🔥 THIS IS KEY

        return {
            "error": str(e)
        }