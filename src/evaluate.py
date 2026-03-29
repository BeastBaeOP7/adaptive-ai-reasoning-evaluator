import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from preprocess import load_data, preprocess_data
from features import create_features
from memory import MistakeMemory

def evaluate_model(df):
    """
    Generate graphs and insights
    """

    # Load trained model
    model = joblib.load("outputs/models/model.pkl")

    # Features
    X = df.drop(columns=[
        'prompt',
        'generated',
        'generated answer',
        'correct answer',
        'label'
    ])

    y = df['label']

    # Predictions
    df['prediction'] = model.predict(X)

    # ---------------------------
    # 1. Accuracy by problem type
    # ---------------------------
    problem_cols = [col for col in df.columns if "problem type_" in col]

    problem_accuracy = {}

    for col in problem_cols:
        subset = df[df[col] == 1]
        if len(subset) > 0:
            acc = (subset['prediction'] == subset['label']).mean()
            problem_accuracy[col.replace("problem type_", "")] = acc

    # Plot
    plt.figure()
    plt.bar(problem_accuracy.keys(), problem_accuracy.values())
    plt.xticks(rotation=45)
    plt.title("Accuracy by Problem Type")
    plt.tight_layout()
    plt.savefig("outputs/graphs/problem_type_accuracy.png")
    plt.show()

    # ---------------------------
    # 2. Reasoning length vs correctness
    # ---------------------------
    plt.figure()
    sns.boxplot(x='label', y='reason_length', data=df)
    plt.title("Reason Length vs Correctness")
    plt.savefig("outputs/graphs/reason_length.png")
    plt.show()

    # ---------------------------
    # 3. Feature importance
    # ---------------------------
    importances = model.feature_importances_
    feature_names = X.columns

    feat_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(10)

    plt.figure()
    sns.barplot(x='importance', y='feature', data=feat_df)
    plt.title("Top 10 Important Features")
    plt.savefig("outputs/graphs/feature_importance.png")
    plt.show()

    print("Evaluation graphs saved in outputs/graphs/ ✅")


if __name__ == "__main__":
    path = "data/nemotron_traj.csv"

    df = load_data(path)
    df = preprocess_data(df)
    df = create_features(df)

    evaluate_model(df)


memory = MistakeMemory()

adjusted_preds = []

for i, row in df.iterrows():
    pred = row['prediction']

    # If wrong → store
    if pred != row['label']:
        memory.store(row)

    # If similar mistake seen → adjust
    similar = memory.check_similar_mistake(row)
    if similar and pred == 1 and row['reason_length'] > 10000:  # If model predicted correct but similar mistake exists
        pred = 0  # Adjust to incorrect

    adjusted_preds.append(pred)

df['adjusted_prediction'] = adjusted_preds

from sklearn.metrics import accuracy_score

# Original accuracy
original_acc = accuracy_score(df['label'], df['prediction'])

# New accuracy after memory correction
adjusted_acc = accuracy_score(df['label'], df['adjusted_prediction'])

print("\nOriginal Accuracy:", original_acc)
print("Adjusted Accuracy:", adjusted_acc)