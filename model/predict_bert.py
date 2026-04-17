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
            MODEL_NAME,
            low_cpu_mem_usage=True,   # 🔥 reduces RAM usage
            torch_dtype=torch.float32 # safer for CPU
        )

        model.to("cpu")  # 🔥 force CPU
        model.eval()

        print("Model loaded successfully!")


def predict(text):
    load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128  # 🔥 IMPORTANT: limits memory usage
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    return {
        "prediction": int(predicted_class_id)
    }