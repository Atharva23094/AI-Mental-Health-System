from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "Atharva233/mental-health-model"

model = None
tokenizer = None


def load_model():
    global model, tokenizer

    if model is None:
        print("🔄 Loading model from HuggingFace...")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # 🔥 IMPORTANT FIXES
        model.to("cpu")  # force CPU (Render safe)
        model.eval()

        torch.set_num_threads(1)  # reduce memory usage

        print("✅ Model loaded successfully!")


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
        predicted_class_id = torch.argmax(logits, dim=1).item()

        # Optional: map labels (edit based on your model)
        label_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        return {
            "text": text,
            "prediction": label_map.get(predicted_class_id, str(predicted_class_id)),
            "class_id": predicted_class_id
        }

    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return {
            "error": str(e)
        }