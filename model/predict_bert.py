from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "Atharva233/mental-health-model"

model = None
tokenizer = None


def load_model():
    global model, tokenizer

    if model is None:
        print("Loading model from HuggingFace...")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        model.eval()

        print("Model loaded successfully!")


def predict(text):
    load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # 🔥 CRITICAL FIX (DistilBERT)
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