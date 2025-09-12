import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os

output_folder = "eda_viz"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df = pd.read_csv("data/SerbMR-2C.csv")
df.rename(columns={'Text': 'text', 'class-att': 'label_text'}, inplace=True)
df['label'] = df['label_text'].apply(lambda x: 'positive' if str(x).lower() == 'positive' else 'negative')

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

def word_count(text):
    return len(text.split())

def char_count(text):
    return len(text)

df['clean_text'] = df['text'].fillna("").apply(clean_text)
df['word_count'] = df['clean_text'].apply(word_count)
df['char_count'] = df['text'].fillna("").apply(char_count)

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='word_count', hue='label', multiple='dodge', bins=50, kde=True, shrink=0.8)
plt.title("Distribucija broja reči po klasama")
plt.xlabel("Broj reči")
plt.ylabel("Broj recenzija")
plt.savefig(os.path.join(output_folder, "eda_wordcount_dodge.png"), dpi=200, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='char_count', hue='label', multiple='dodge', bins=50, kde=True, shrink=0.8)
plt.title("Distribucija broja karaktera po klasama")
plt.xlabel("Broj karaktera")
plt.ylabel("Broj recenzija")
plt.savefig(os.path.join(output_folder, "eda_charcount_dodge.png"), dpi=200, bbox_inches='tight')
plt.close()

print(f"Plot saved in '{output_folder}'.")