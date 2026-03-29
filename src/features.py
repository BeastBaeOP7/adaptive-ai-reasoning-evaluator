import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# global vectorizer (IMPORTANT)
tfidf = None


def create_features(df, fit=True, vectorizer=None):
    """
    Create numerical + TF-IDF features
    """

    # ---------------------------
    # 1. Basic features
    # ---------------------------
    df['reason_length'] = df['generated'].apply(len)
    df['prompt_length'] = df['prompt'].apply(len)
    df['word_count'] = df['generated'].apply(lambda x: len(x.split()))
    df['digit_count'] = df['generated'].str.count(r'\d')

    # ---------------------------
    # 2. Encode problem type
    # ---------------------------
    df = pd.get_dummies(df, columns=['problem type'])

    # ---------------------------
    # 3. TF-IDF (🔥 NEW)
    # ---------------------------
    if fit:
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(df['generated'])
    else:
        tfidf

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    )

    # Reset index before concat
    df = df.reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)

    df = pd.concat([df, tfidf_df], axis=1)

    return df, vectorizer




if __name__ == "__main__":
    from preprocess import load_data, preprocess_data

    path = "data/nemotron_traj.csv"

    df = load_data(path)
    df = preprocess_data(df)
    df = create_features(df)

    print("Feature Engineering Done ✅")
    print(df.head())
    print("\nColumns:\n", df.columns)