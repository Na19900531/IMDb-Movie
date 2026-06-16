import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


df = pd.read_csv("movies.csv")

def clean_text(text):
    if pd.isnull(text):
        return ""
  
    text = text.lower()
  
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


df["clean_storyline"] = df["Storyline"].apply(clean_text)


df = df[df["clean_storyline"].str.strip() != ""]

df.to_csv("movies_cleaned.csv", index=False)

print("✅ Preprocessing complete. Cleaned file saved as movies_cleaned.csv")
print(df.head())
