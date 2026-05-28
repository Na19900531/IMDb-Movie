from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

X = vectorizer.fit_transform("imdb_recommender\movies.csv")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('movies.csv')

df['Cleaned_Storyline'] = df['Cleaned_Storyline'].fillna('').astype(str)


df = df[df['Cleaned_Storyline'].str.strip() != '']


vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")


X = vectorizer.fit_transform(df['Cleaned_Storyline'])

print("Success! Vocabulary size:", len(vectorizer.vocabulary_))
