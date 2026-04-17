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

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,      # CPU safe
            low_cpu_mem_usage=True          # 🔥 reduces memory usage
        )

        model.to("cpu")
        model.eval()

        # Reduce CPU threads (important for Render)
        torch.set_num_threads(1)

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

        return {
            "prediction": predicted_class_id
        }

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return {
            "error": str(e)
        }