from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = 0 if device.type == 'cuda' else -1
print(f"Using device: {device}")

def analyze_sentiment(text, model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    
    prediction = torch.argmax(logits, dim=1).item()
    score = torch.softmax(logits, dim=1)[0][prediction].item()

    label_mapping = {0: 'negative', 1: 'positive'}
    mapped_label = label_mapping.get(prediction, 'unknown')

    return {'label': mapped_label, 'score': score} 

if __name__ == "__main__":
    texts_to_analyze = [
        "Ovaj film je bio fantastičan, preporučujem ga svima!",
        "Hrana u restoranu je bila grozna, nikada se više neću vratiti.",
        "Proizvod je stigao na vrijeme i radi kako treba. Zadovoljan sam kupovinom.",
        "Iako je bilo nekih problema, na kraju smo uspješno riješili sve nedoumice.",
        "Ovo je najgora usluga koju sam ikada doživio.",
    ]

    hf_username = "srdjoo14"

    # If you want to use models that you are trained from the start =>
    # model_path_bertic = "./models/classla_bcms-bertic"
    # model_path_mbert = "./models/google-bert_bert-base-multilingual-cased"

    # Test for the first model - bertic
    model_path_bertic = f"{hf_username}/serbian-sentiment-bertic"
    for text in texts_to_analyze:
        result = analyze_sentiment(text, model_path_bertic, device)
        print(f"BERTIC - text: '{text}' -> Result: {result}\n")
    
    print("-" * 50)
    
    # Test for the second model - mBERT
    model_path_mbert = f"{hf_username}/serbian-sentiment-mbert"
    for text in texts_to_analyze:
        result = analyze_sentiment(text, model_path_mbert, device)
        print(f"MBERT - text: '{text}' -> Result: {result}\n")