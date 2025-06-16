import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import dagshub

# === Konfigurasi DagsHub MLflow dengan default access token ===
dagshub.init(repo_owner='hara29', repo_name='bank-attrition-modelling', mlflow=True)
mlflow.set_experiment("RandomForest_ManualLogging")

# Load data
X_train = pd.read_csv("bank_preprocessing/X_train.csv")
y_train = pd.read_csv("bank_preprocessing/y_train.csv").values.ravel()
X_test = pd.read_csv("bank_preprocessing/X_test.csv")
y_test = pd.read_csv("bank_preprocessing/y_test.csv").values.ravel()
input_example = X_train[0:5]

with mlflow.start_run():
    # Parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Predict
    y_pred = best_model.predict(X_test)

    # Manual logging
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # Simpan model dengan signature input
    mlflow.sklearn.log_model(best_model, "model", input_example=input_example)

    # Log confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    with tempfile.NamedTemporaryFile(suffix="_cm.png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        mlflow.log_artifact(tmpfile.name, artifact_path="plots")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
