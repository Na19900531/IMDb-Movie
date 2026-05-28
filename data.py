import pandas as pd

df = pd.read_csv("imdb_movies_2024.csv")


print(df.head())
print("Missing storylines:", df["Storyline"].isna().sum())
print("Empty storylines:", (df["Storyline"].str.strip() == "").sum())


df = df.dropna(subset=["Storyline"])              
df = df[df["Storyline"].str.strip() != ""]       


df.to_csv("movies_clean.csv", index=False)
print("✅ Cleaned data saved to movies_clean.csv")


df = pd.read_csv("movies_clean.csv")
print(df.head())

df = pd.read_csv("C:/Users/Welcome/Desktop/imdb_recommender/imdb_movies_2024.csv")
df = pd.read_csv("movies.csv")

df = df.dropna(subset=["Storyline"])
df = df[df["Storyline"].str.strip() != ""]
df.to_csv("movies_clean.csv", index=False)
print("✅ Cleaned data saved to movies_clean.csv")
