import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from xgboost import XGBClassifier

from preprocess import load_data, preprocess_data
from features import create_features


def train_model(df):
    """
    Train ML model using XGBoost
    """

    # ---------------------------
    # 1. Select features
    # ---------------------------
    X = df.drop(columns=[
        'prompt',
        'generated',
        'generated answer',
        'correct answer',
        'label'
    ])

    y = df['label']

    # ---------------------------
    # 2. Train-test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # 3. Train model (🔥 XGBoost)
    # ---------------------------
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # ---------------------------
    # 4. Save feature columns
    # ---------------------------
    feature_cols = X.columns
    joblib.dump(feature_cols, "outputs/models/feature_cols.pkl")

    # ---------------------------
    # 5. Predictions
    # ---------------------------
    y_pred = model.predict(X_test)

    # ---------------------------
    # 6. Evaluation
    # ---------------------------
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    path = "data/nemotron_traj.csv"

    df = load_data(path)
    df = preprocess_data(df)
    df, vectorizer = create_features(df, fit=True)

    model = train_model(df)

    # ---------------------------
    # 7. Save model
    # ---------------------------
    joblib.dump(model, "outputs/models/model.pkl")
    joblib.dump(vectorizer, "outputs/models/tfidf.pkl")
    print("\nModel + feature columns saved in outputs/models/ ✅")