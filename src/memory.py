from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MistakeMemory:
    def __init__(self):
        self.texts = []
        self.vectorizer = TfidfVectorizer(max_features=300)
        self.matrix = None

    def store(self, row):
        self.texts.append(row["generated"])

        # Recompute vector matrix occasionally
        if len(self.texts) % 50 == 0:
            self.matrix = self.vectorizer.fit_transform(self.texts)

    def check_similar_mistake(self, row):
        if self.matrix is None or len(self.texts) < 50:
            return False

        new_vec = self.vectorizer.transform([row["generated"]])
        sim = cosine_similarity(new_vec, self.matrix)

        return sim.max() > 0.75