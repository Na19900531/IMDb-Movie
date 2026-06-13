from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["Storyline"])
    return vectorizer, tfidf_matrix

def recommend_movies(user_storyline, df, vectorizer, tfidf_matrix, top_n=5):
    user_vec = vectorizer.transform([user_storyline])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][["Movie Name", "Storyline"]]