from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

MODEL_NAME = "Atharva233/mental-health-model"

model = None
tokenizer = None


def load_model():
    global model, tokenizer

    if model is None:
        print("Loading model...")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32  # SAFE for CPU
        )

        model.to("cpu")
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

        # ❗ REMOVE token_type_ids (important for DistilBERT)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

        return {
            "prediction": pred,
            "confidence": round(conf, 4)
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "error": str(e)
        }