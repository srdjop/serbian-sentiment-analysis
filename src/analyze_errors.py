# analyze_errors.py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

hf_username = "srdjoo14"

model_paths = {
    "bertic": f"{hf_username}/serbian-sentiment-bertic",
    "mbert": f"{hf_username}/serbian-sentiment-mbert"
}

RESULTS_DIR = Path("eval_results")
OUT_DIR = Path("error_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Učitaj dataset (isti kao u evaluaciji)
df = pd.read_csv("data/SerbMR-2C.csv")
df.rename(columns={'Text': 'text', 'class-att': 'label_text'}, inplace=True)
df['label'] = df['label_text'].apply(lambda x: 1 if str(x).lower() == 'positive' else 0)
df_final = df[['text', 'label']]

# Podela train/test kao ranije (da bi se poklapalo sa evaluacijom)
from datasets import Dataset, ClassLabel
hf_dataset = Dataset.from_pandas(df_final)
class_label_object = ClassLabel(num_classes=2, names=['negative', 'positive'])
hf_dataset = hf_dataset.cast_column('label', class_label_object)
train_test_split = hf_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
test_set = train_test_split['test']

# Izvlačenje grešaka
def analyze_model(model_name, model_path):
    print(f"\n--- Analiza grešaka: {model_name} ---")

    # Učitavanje tokenizer-a i modela
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts, y_true, y_pred, confidences = [], [], [], []

    with torch.no_grad():
        for example in test_set:
            text = example["text"]
            label = example["label"]

            enc = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            outputs = model(**enc)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

            pred = int(probs.argmax())
            conf = float(probs[pred])

            texts.append(text)
            y_true.append(label)
            y_pred.append(pred)
            confidences.append(conf)

    df_res = pd.DataFrame({
        "text": texts,
        "true_label": y_true,
        "pred_label": y_pred,
        "confidence": confidences
    })

    # Filtriraj samo greške
    df_errors = df_res[df_res["true_label"] != df_res["pred_label"]]

    # Sortiraj po najvišoj konfidenciji → "najubedljivije greške"
    df_top_errors = df_errors.sort_values(by="confidence", ascending=False).head(10)

    # Sačuvaj fajlove
    out_model_dir = OUT_DIR / model_name
    out_model_dir.mkdir(parents=True, exist_ok=True)

    df_errors.to_csv(out_model_dir / "all_errors.csv", index=False, encoding="utf-8")
    df_top_errors.to_csv(out_model_dir / "top10_errors.csv", index=False, encoding="utf-8")

    print(f"[INFO] Sačuvano {len(df_errors)} grešaka, top 10 u: {out_model_dir/'top10_errors.csv'}")

if __name__ == "__main__":
    for model_name, model_path in model_paths.items():
        analyze_model(model_name, model_path)
