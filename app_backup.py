import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

st.set_page_config(
    page_title="AI Model Comparison Tool",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Model Comparison Tool")
st.write(
    "Train, evaluate, and compare machine learning models "
    "using a consistent dataset and evaluation process."
)

# Sidebar settings
st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider(
    "Test data percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
)

tree_depth = st.sidebar.slider(
    "Decision Tree maximum depth",
    min_value=1,
    max_value=10,
    value=3,
)

forest_trees = st.sidebar.slider(
    "Random Forest number of trees",
    min_value=10,
    max_value=200,
    value=100,
    step=10,
)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=42,
    stratify=y,
)

# Define models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=tree_depth,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=forest_trees,
        random_state=42,
    ),
}

results = []
trained_models = {}

# Train and evaluate
for name, model in models.items():
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_ms = (time.perf_counter() - start_train) * 1000

    start_predict = time.perf_counter()
    predictions = model.predict(X_test)
    predict_ms = (time.perf_counter() - start_predict) * 1000

    trained_models[name] = model

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions,
            average="weighted",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test, predictions,
            average="weighted",
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_test, predictions,
            average="weighted",
            zero_division=0,
        ),
        "Training Time (ms)": train_ms,
        "Prediction Time (ms)": predict_ms,
    })

results_df = pd.DataFrame(results)

# Main results
st.subheader("📊 Model Performance")

metric = st.selectbox(
    "Rank models by:",
    ["Accuracy", "Precision", "Recall", "F1 Score"],
)

best_row = results_df.loc[results_df[metric].idxmax()]

col1, col2, col3 = st.columns(3)

col1.metric("Best Model", best_row["Model"])
col2.metric("Best " + metric, f"{best_row[metric]:.2%}")
col3.metric("Test Samples", len(y_test))

st.subheader("📋 Detailed Comparison")

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}",
        "Training Time (ms)": "{:.2f}",
        "Prediction Time (ms)": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# Performance chart
st.subheader("📈 Metric Comparison")

chart_data = results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
]

st.bar_chart(chart_data)

# Time comparison
st.subheader("⏱️ Execution Time Comparison")

time_data = results_df.set_index("Model")[
    ["Training Time (ms)", "Prediction Time (ms)"]
]

st.bar_chart(time_data)

# Confusion matrix
st.subheader("🎯 Confusion Matrix")

selected_name = st.selectbox(
    "Choose a model to inspect:",
    list(trained_models.keys()),
)

selected_model = trained_models[selected_name]
predictions = selected_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2],
)

fig, ax = plt.subplots(figsize=(6, 4))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names,
).plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
)

ax.set_title(selected_name)
st.pyplot(fig)
plt.close(fig)

# Feature importance
if selected_name in ["Decision Tree", "Random Forest"]:
    st.subheader("🔍 Feature Importance")

    importance = selected_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": iris.feature_names,
        "Importance": importance,
    }).sort_values("Importance", ascending=False)

    st.bar_chart(
        importance_df.set_index("Feature")
    )

st.caption(
    "Dataset: Iris | Split: stratified train/test | "
    "Timing measurements may vary between runs and devices."
)
# Live prediction
st.divider()
st.subheader("🌸 Try a New Prediction")

st.write(
    "Enter flower measurements and let the selected "
    "machine learning model predict the flower species."
)

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.1,
        max_value=10.0,
        value=5.1,
        step=0.1,
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.1,
        max_value=10.0,
        value=3.5,
        step=0.1,
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.1,
        max_value=10.0,
        value=1.4,
        step=0.1,
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=10.0,
        value=0.2,
        step=0.1,
    )

prediction_model_name = st.selectbox(
    "Select a prediction model:",
    list(trained_models.keys()),
    key="prediction_model",
)

if st.button("🔮 Predict Flower Species"):
    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
        ]],
        columns=iris.feature_names,
    )

    model = trained_models[prediction_model_name]
    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Species: {iris.target_names[prediction].title()}"
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        probability_df = pd.DataFrame({
            "Species": [
                name.title() for name in iris.target_names
            ],
            "Probability": probabilities,
        }).set_index("Species")

        st.write("Prediction probabilities:")
        st.bar_chart(probability_df)