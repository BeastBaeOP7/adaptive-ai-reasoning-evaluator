import pandas as pd


def load_data(path):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(path)
    return df


def clean_text(text):
    """
    Basic text cleaning
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    return text


def preprocess_data(df):
    """
    Full preprocessing pipeline
    """

    # ---------------------------
    # 1. Handle missing values
    # ---------------------------
    df['generated answer'] = df['generated answer'].fillna("UNKNOWN")

    # ---------------------------
    # 2. Clean text columns
    # ---------------------------
    df['prompt'] = df['prompt'].apply(clean_text)
    df['generated'] = df['generated'].apply(clean_text)
    df['generated answer'] = df['generated answer'].apply(clean_text)
    df['correct answer'] = df['correct answer'].apply(clean_text)

    # ---------------------------
    # 3. Create target label
    # ---------------------------
    # true -> 1, false/partial -> 0
    df['label'] = df['correctness'].apply(
        lambda x: 1 if x == 'true' else 0
    )

    # ---------------------------
    # 4. Drop unnecessary columns (optional)
    # ---------------------------
    # Keep useful columns only
    df = df[
        [
            'prompt',
            'generated',
            'generated answer',
            'correct answer',
            'problem type',
            'label'
        ]
    ]

    return df


if __name__ == "__main__":
    # Test run
    path = "data/nemotron_traj.csv"

    df = load_data(path)
    df = preprocess_data(df)

    print("Preprocessing Done ✅")
    print(df.head())
    print("\nShape:", df.shape)