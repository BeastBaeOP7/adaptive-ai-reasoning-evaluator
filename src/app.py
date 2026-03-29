import pandas as pd
import joblib

from preprocess import preprocess_data
from features import create_features
from memory import MistakeMemory


# ---------------------------
# Load model + feature schema
# ---------------------------
model = joblib.load("outputs/models/model.pkl")
feature_cols = joblib.load("outputs/models/features_cols.pkl")

memory = MistakeMemory()


# ---------------------------
# Prepare input
# ---------------------------
def prepare_input(prompt, generated, gen_ans, correct_ans, problem_type):

    data = {
        "prompt": [prompt],
        "generated": [generated],
        "generated answer": [gen_ans],
        "correct answer": [correct_ans],
        "problem type": [problem_type],
        "correctness": ["false"]  # dummy label
    }

    df = pd.DataFrame(data)

    # Preprocess + feature engineering
    df = preprocess_data(df)
    df = create_features(df, fit=False)

    # Extract features
    X = df.drop(columns=[
        'prompt',
        'generated',
        'generated answer',
        'correct answer',
        'label'
    ])

    # Align features with training
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols]

    return df, X


# ---------------------------
# Main app loop
# ---------------------------
def main():
    print("🧠 AI Reasoning Evaluator (Interactive Mode)\n")

    while True:
        print("\n--- New Input ---")

        prompt = input("Enter prompt: ")
        generated = input("Enter reasoning: ")
        gen_ans = input("Generated answer: ")
        correct_ans = input("Correct answer (optional): ")
        problem_type = input("Problem type (e.g., gravity, cipher): ")

        df, X = prepare_input(prompt, generated, gen_ans, correct_ans, problem_type)

        # Prediction + confidence
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][pred]

        print("\nModel Prediction:",
              "✅ Correct" if pred == 1 else "❌ Incorrect",
              f"(Confidence: {prob:.2f})")

        # Memory-based adjustment
        similar = memory.check_similar_mistake(df.iloc[0])

        if pred == 1 and similar:
            pred = 0
            print("⚠️ Adjusted due to similar past mistake")

        # Store mistake
        if pred != df.iloc[0]['label']:
            memory.store(df.iloc[0])

        print("Final Prediction:",
              "✅ Correct" if pred == 1 else "❌ Incorrect")

        cont = input("\nContinue? (y/n): ")
        if cont.lower() != 'y':
            break


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    main()