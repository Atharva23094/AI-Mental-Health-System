from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

MODEL_NAME = "Atharva233/mental-health-model"

model = None
tokenizer = None


def load_model():
    global model, tokenizer

    if model is None:
        print("Loading model from HuggingFace...")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        model.eval()
        print("Model loaded successfully!")


def predict(text):
    try:
        load_model()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        predicted_class_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class_id].item()

        return {
            "prediction": predicted_class_id,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "error": str(e)
        }