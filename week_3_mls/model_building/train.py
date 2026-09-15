import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import mlflow

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Load data
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# Features
numeric_features = ["Air temperature", "Process temperature",
                    "Rotational speed", "Torque", "Tool wear"]
categorical_features = ["Type"]

# Handle imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

# Start main MLflow run
parent_run = mlflow.start_run()

# Grid search
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1
)
grid_search.fit(Xtrain, ytrain)

# Log each param set
results = grid_search.cv_results_
for i in range(len(results["params"])):
    with mlflow.start_run(run_id=parent_run.info.run_id, nested=True):
        mlflow.log_params(results["params"][i])
        mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
        mlflow.log_metric("std_test_score", results["std_test_score"][i])

# Log best params
mlflow.log_params(grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Evaluate
threshold = 0.45
y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)
y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)

train_report = classification_report(ytrain, y_pred_train, output_dict=True)
test_report = classification_report(ytest, y_pred_test, output_dict=True)

mlflow.log_metrics({
    "train_accuracy": train_report["accuracy"],
    "train_precision": train_report["1"]["precision"],
    "train_recall": train_report["1"]["recall"],
    "train_f1": train_report["1"]["f1-score"],
    "test_accuracy": test_report["accuracy"],
    "test_precision": test_report["1"]["precision"],
    "test_recall": test_report["1"]["recall"],
    "test_f1": test_report["1"]["f1-score"]
})

# Save model
model_path = "week_3_mls/deployment/best_machine_failure_model_v1.joblib"
joblib.dump(best_model, model_path)
mlflow.log_artifact(model_path, artifact_path="model")

print(f"Model saved to {model_path}")

mlflow.end_run()
