import time

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define the models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
}

# 4. Train and evaluate each model
results = []

for name, model in models.items():
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - start_predict

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Training Time (ms)": train_time * 1000,
        "Prediction Time (ms)": predict_time * 1000,
    })

# 5. Display comparison results
results_df = pd.DataFrame(results)

print("\n=== AI Model Comparison Tool ===")
print("Dataset: Iris")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("\nModel Performance:")
print(results_df.round(4).to_string(index=False))

best_model = results_df.loc[results_df["F1 Score"].idxmax()]
print(f"\nBest model by F1 Score: {best_model['Model']}")
print(f"F1 Score: {best_model['F1 Score']:.4f}")
