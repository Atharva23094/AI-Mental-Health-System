from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "Atharva233/mental-health-model"

model = None
tokenizer = None


def load_model():
    global model, tokenizer
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()


def predict(text):
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
    predicted_class_id = torch.argmax(logits, dim=1).item()

    return predicted_class_id