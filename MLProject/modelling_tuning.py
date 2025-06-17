import os
import json
import pandas as pd
import mlflow
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models.signature import infer_signature
from mlflow.sklearn import log_model
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load data
    X_train = pd.read_csv("bank_preprocessing/X_train.csv")
    y_train = pd.read_csv("bank_preprocessing/y_train.csv").values.ravel()
    X_test = pd.read_csv("bank_preprocessing/X_test.csv")
    y_test = pd.read_csv("bank_preprocessing/y_test.csv").values.ravel()

    with mlflow.start_run():
        # Parameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        y_pred = best_model.predict(X_test)

        # Log parameter dan metrik ke MLflow
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        })

        # Log model
        log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_test.iloc[:5].astype("float64"),
            signature=infer_signature(X_test.astype("float64"), y_pred)
        )

        # Simpan confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("training_confusion_matrix.png")
        mlflow.log_artifact("training_confusion_matrix.png")

        # Simpan metric_info.json
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        with open("metric_info.json", "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact("metric_info.json")

        # Simpan estimator.html
        with open("estimator.html", "w") as f:
            f.write(f"""
            <html>
            <head><title>Estimator Summary</title></head>
            <body>
            <h2>Best Estimator</h2>
            <pre>{best_model}</pre>

            <h3>Best Parameters</h3>
            <ul>
                <li>n_estimators: {best_params.get('n_estimators')}</li>
                <li>max_depth: {best_params.get('max_depth')}</li>
                <li>min_samples_split: {best_params.get('min_samples_split')}</li>
            </ul>

            <h3>Best Cross-Validation Score (F1)</h3>
            <p>{best_score:.4f}</p>

            <h3>Grid Search Config</h3>
            <ul>
                <li>cv: 3-fold</li>
                <li>scoring: f1</li>
                <li>random_state: 42</li>
            </ul>
            </body>
            </html>
            """)
        mlflow.log_artifact("estimator.html")

        # Print klasifikasi akhir
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
