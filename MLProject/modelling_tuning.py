
import os
import json
import pandas as pd
import mlflow
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Konfigurasi DagsHub MLflow ===
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri("https://dagshub.com/hara29/bank-attrition-modelling.mlflow")
mlflow.set_experiment("RandomForest_ManualLogging")

# Load data
X_train = pd.read_csv("bank_preprocessing/X_train.csv")
y_train = pd.read_csv("bank_preprocessing/y_train.csv").values.ravel()
X_test = pd.read_csv("bank_preprocessing/X_test.csv")
y_test = pd.read_csv("bank_preprocessing/y_test.csv").values.ravel()

with mlflow.start_run() as run:
    # Parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Predict
    y_pred = best_model.predict(X_test)

    # Logging parameters & metrics
    mlflow.log_params(best_params)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model (as MLflow model)
    import mlflow.sklearn
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=X_test.iloc[:5].astype("float64"),
        signature=infer_signature(X_test.astype("float64"), y_pred)
    )

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_plot_path = "training_confusion_matrix.png"
    plt.savefig(cm_plot_path)

    # Save additional metrics and estimator info
    with open("metric_info.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }, f, indent=2)

    with open("estimator.html", "w") as f:
        html = """<html>
        <head><title>Estimator Summary</title></head>
        <body>
        <h2>Best Estimator</h2>
        <pre>{}</pre>
        </body>
        </html>""".format(best_model)
        f.write(html)

    mlflow.log_artifact(cm_plot_path)
    mlflow.log_artifact("metric_info.json")
    mlflow.log_artifact("estimator.html")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
