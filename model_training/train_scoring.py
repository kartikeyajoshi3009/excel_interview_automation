import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import joblib


# Load your dataset
df = pd.read_json(r"C:\Users\karti\OneDrive\Desktop\excel interview\dataset\excel_qa_augmented_dynamic_with_scores_structured.json")

# Combine question, answer and type into one text feature
df["text"] = df["question"] + " " + df["answer"] + " " + df["type"]

# Features and multi-output targets
X = df["text"]
y = df[["correctness", "clarity", "terminology", "efficiency"]]

# Vectorize text with TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X_vec = vectorizer.fit_transform(X)

# Split 80-20 train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# MultiOutput Ridge Regression Model
reg = MultiOutputRegressor(Ridge())
reg.fit(X_train, y_train)

# Predict on train and test set
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Print error metrics per score
def print_metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    metrics_df = pd.DataFrame({
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    }, index=["correctness", "clarity", "terminology", "efficiency"])
    print(f"\nError metrics for {label} data:")
    print(metrics_df)

print_metrics(y_train, y_train_pred, "training")
print_metrics(y_test, y_test_pred, "testing")

# Overall R2 scores as accuracy proxy for train/test
print(f"\nOverall training R2 score: {r2_score(y_train, y_train_pred)}")
print(f"Overall testing R2 score: {r2_score(y_test, y_test_pred)}")

# Save model and vectorizer
joblib.dump(reg, "ridge_multioutput_regressor_updated_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer_updated_model.joblib")
print("\nTrained model and vectorizer saved successfully.")

# --- Begin plotting and evaluation (unchanged) ---

# Print error metrics for testing data metrics_df for reference (optional)
print("\nError metrics per score on testing data:")
print_metrics(y_test, y_test_pred, "testing data")

# Scatter plots: true vs predicted scores on test data
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
score_names = ["correctness", "clarity", "terminology", "efficiency"]

for i, ax in enumerate(axs.flatten()):
    ax.scatter(y_test.iloc[:, i], y_test_pred[:, i], alpha=0.5)
    ax.plot([0, 10], [0, 10], "r--")  # Ideal line
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"True vs Predicted - {score_names[i]}")

plt.tight_layout()
plt.show()

# Residual histograms on test data
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for i, ax in enumerate(axs.flatten()):
    residuals = y_test.iloc[:, i] - y_test_pred[:, i]
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title(f"Residuals - {score_names[i]}")
    ax.set_xlabel("True - Predicted")

plt.tight_layout()
plt.show()

# Discretize continuous scores for confusion matrices
y_test_discrete = np.clip(np.rint(y_test.values), 0, 10).astype(int)
y_pred_discrete = np.clip(np.rint(y_test_pred), 0, 10).astype(int)

labels = np.arange(0, 11)

# Heatmap - Confusion Matrix with counts on test
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
for i, ax in enumerate(axs.flatten()):
    cm = confusion_matrix(y_test_discrete[:, i], y_pred_discrete[:, i], labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Prediction Counts Heatmap - {score_names[i]}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.show()

# Heatmap - Confusion Matrix normalized by true label (percentage)
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
for i, ax in enumerate(axs.flatten()):
    cm_norm = confusion_matrix(y_test_discrete[:, i], y_pred_discrete[:, i], labels=labels, normalize="true")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title(f"Prediction Percentage Heatmap - {score_names[i]}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.show()
