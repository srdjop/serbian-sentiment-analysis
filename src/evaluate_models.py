import pandas as pd
import os
from pathlib import Path
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
import numpy as np

hf_username = "srdjoo14"

model_paths = {
    "bertic": f"{hf_username}/serbian-sentiment-bertic",
    "mbert": f"{hf_username}/serbian-sentiment-mbert"
}

# STEP 1: Loading test set
try:
    df = pd.read_csv("data/SerbMR-2C.csv")
    df.rename(columns={'Text': 'text', 'class-att': 'label_text'}, inplace=True)
    df['label'] = df['label_text'].apply(lambda x: 1 if str(x).lower() == 'positive' else 0)
    df_final = df[['text', 'label']]
except Exception as e:
    print(f"Error while loading the data: {e}")
    exit()

hf_dataset = Dataset.from_pandas(df_final)
class_label_object = ClassLabel(num_classes=2, names=['negative', 'positive'])
hf_dataset = hf_dataset.cast_column('label', class_label_object)
train_test_split = hf_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
test_set = train_test_split['test']

print("Starting model evaluation...")

RESULTS_DIR = Path("eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for model_name, model_path in model_paths.items():
    print(f"\n--- Model evaluation: {model_name} ---")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
        
        tokenized_test_set = test_set.map(tokenize_function, batched=True)
        tokenized_test_set = tokenized_test_set.remove_columns(['text'])

        model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for example in tokenized_test_set:
                input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
                attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)
                labels = example['label']

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).cpu().item()
                
                predictions.append(pred)
                true_labels.append(labels)
        
        # Metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        report = classification_report(true_labels, predictions, target_names=['negative', 'positive'], digits=4)
        cm = confusion_matrix(true_labels, predictions)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:\n", report)
        print("\nConfusion Matrix:\n", cm)

        model_dir = RESULTS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # CSV with predictions
        pd.DataFrame({
            "text": test_set["text"],
            "true_label": true_labels,
            "pred_label": predictions
        }).to_csv(model_dir / "predictions.csv", index=False, encoding="utf-8")

        # Report
        with open(model_dir / "report.txt", "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(cm))

        # NumPy files for plotting
        np.save(model_dir / "y_true.npy", np.array(true_labels))
        np.save(model_dir / "y_pred.npy", np.array(predictions))
        np.save(model_dir / "confusion_matrix.npy", cm)

        print(f"\nResults saved in folder: {model_dir}")

    except Exception as e:
        print(f"Error in model evaluation '{model_name}': {e}")
        continue
