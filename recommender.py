import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, csv_file="movies.csv"):
        self.df = pd.read_csv(csv_file)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["Storyline"])

    def recommend(self, input_storyline, top_n=5):
        input_vec = self.vectorizer.transform([input_storyline])
        similarity_scores = cosine_similarity(input_vec, self.tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]

        recommendations = self.df.iloc[top_indices][["Movie Name", "Storyline"]]
        return recommendations

if __name__ == "__main__":
    recommender = MovieRecommender()
    test_story = "A young wizard begins his journey at a magical school."
    print(recommender.recommend(test_story))
