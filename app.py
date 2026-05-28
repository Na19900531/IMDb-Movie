import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data(csv_file="movies_clean.csv"):
    try:
        df = pd.read_csv(csv_file)

        # Ensure correct column names
        if "Title" in df.columns and "Plot" in df.columns:
            df.rename(columns={"Title": "Movie Name", "Plot": "Storyline"}, inplace=True)

        df = df.dropna(subset=["Storyline"])
        df = df[df["Storyline"].str.strip() != ""]

        if df.empty:
            raise ValueError("Dataset has no valid storylines. Please check your CSV file.")

        return df
    except FileNotFoundError:
        st.error(f"File {csv_file} not found. Please place it in the same folder as app.py.")
        st.stop()


class MovieRecommender:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["Storyline"])

    def recommend(self, input_storyline, top_n=5):
        input_vec = self.vectorizer.transform([input_storyline])
        similarity_scores = cosine_similarity(input_vec, self.tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        return self.df.iloc[top_indices][["Movie Name", "Storyline"]]


st.title("🎬 IMDb Movie Recommendation System")
st.write("Enter a movie storyline and get recommendations!")

df = load_data("movies_clean.csv")

@st.cache_resource
def get_recommender(df):
    return MovieRecommender(df)

recommender = get_recommender(df)

user_input = st.text_area("Enter storyline:", "")

if st.button("Recommend"):
    if not user_input.strip():
        st.warning("Please enter a storyline.")
    else:
        results = recommender.recommend(user_input)
        st.subheader("Top 5 Recommended Movies:")
        for _, row in results.iterrows():
            st.write(f"**{row['Movie Name']}**")
            st.write(row["Storyline"])
            st.write("---")






