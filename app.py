import streamlit as st
from data_loader import load_data
from recommender import build_model, recommend_movies

# Load dataset
csv_path = r"C:\Users\Welcome\Downloads\imdb_movies_2024.csv"
df = load_data(csv_path)

# Build model
vectorizer, tfidf_matrix = build_model(df)

# Streamlit UI
st.title("🎬 IMDb Movie Recommendation System")

user_input = st.text_area("Enter a movie storyline:")
if st.button("Recommend"):
    results = recommend_movies(user_input, df, vectorizer, tfidf_matrix)
    for _, row in results.iterrows():
        st.subheader(row["Movie Name"])
        st.write(row["Storyline"])





