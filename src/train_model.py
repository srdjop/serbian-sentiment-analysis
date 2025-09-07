import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import argparse 

# Adding parser for arguments
parser = argparse.ArgumentParser(description="Training model for sentiment analysis.")
parser.add_argument("--model_name", type=str, default="classla/bcms-bertic", help="Name of Hugging Face model for training.")
args = parser.parse_args()

# Checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# STEP 1: Data loading and preprocessing
try:
    df = pd.read_csv("data/SerbMR-2C.csv")
    df.rename(columns={'Text': 'text', 'class-att': 'label_text'}, inplace=True)
    df['label'] = df['label_text'].apply(lambda x: 1 if str(x).lower() == 'positive' else 0)
    df_final = df[['text', 'label']]
    print("Initial dataset:", list(df.columns))
    print("Final dataset:", list(df_final))
    print(df.head(5))
    print(df_final.head(5))
except Exception as e:
    print(f"Error while data loading / preprocessing: {e}")
    exit()

# Hugging Face Dataset format
hf_dataset = Dataset.from_pandas(df_final)
class_label_object = ClassLabel(num_classes=2, names=['negative', 'positive'])
hf_dataset = hf_dataset.cast_column('label', class_label_object)
train_test_split = hf_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# STEP 2: Tokenization
model_name = args.model_name 
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])

# STEP 3: Model, arguments i training definition
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1}

training_args = TrainingArguments(
    num_train_epochs=5,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    weight_decay=0.1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="steps",
    eval_steps=len(dataset_dict['train']) // 4,
    save_steps=len(dataset_dict['train']) // 4,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# STEP 4: Training
print("Training is started...")
trainer.train()
print("Training is finished!")

# STEP 5: Save model
model_path = f"./models/{model_name.replace('/', '_')}"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model and tokenizer saved in: {model_path}")

