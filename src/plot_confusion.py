import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

RESULTS_DIR = Path("eval_results")
OUT_DIR = Path("confusion_matrices")
OUT_DIR.mkdir(parents=True, exist_ok=True)

labels = ["negative", "positive"]

def plot_and_save_cm(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prediction")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – {model_name}")
    out_path = OUT_DIR / f"{model_name}_cm.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"[INFO] Confusion matrix saved in: {out_path}")

def evaluate_model(model_dir: Path):
    model_name = model_dir.name
    y_true_path = model_dir / "y_true.npy"
    y_pred_path = model_dir / "y_pred.npy"

    if not (y_true_path.exists() and y_pred_path.exists()):
        print(f"[WARN] Skipping {model_name} – there is no y_true/y_pred files for this model.")
        return

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plot_and_save_cm(cm, model_name)

    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    out_txt = OUT_DIR / f"{model_name}_report.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Saved classification report in: {out_txt}")

if __name__ == "__main__":
    for model_dir in RESULTS_DIR.iterdir():
        if model_dir.is_dir():
            evaluate_model(model_dir)
