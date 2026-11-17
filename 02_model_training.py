"""
02_model_training.py
Trains and compares regression and classification models.
Preprocessing is fitted on training data only to reduce leakage.
Run:
    python 02_model_training.py
"""

from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "outputs" / "data" / "cleaned_campaign_data.csv"
MODEL_DIR = ROOT / "outputs" / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_REG = "Revenue"
TARGET_CLS = "Profit_Flag"

CATEGORICAL_FEATURES = [
    "Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment"
]
NUMERIC_FEATURES = [
    "Duration", "Impressions", "Clicks", "Leads", "Conversions",
    "Acquisition_Cost", "Engagement_Score", "CTR", "Conversion_Rate",
    "Cost_per_Conversion", "Campaign_Month", "Campaign_DayOfWeek"
]
EXCLUDED = {
    "Campaign_ID", "Date", "Channel_Used", "Revenue", "ROI", "ROI_Original",
    "ROI_Calculated", "Total_Spend", "Profit_Flag"
}

def make_preprocessor(numeric_features, categorical_features, channel_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
            ("channels", "passthrough", channel_features)
        ],
        remainder="drop",
        sparse_threshold=0.0
    )

def main():
    start = time.time()
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned data not found. Run 01_data_cleaning_eda.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET_REG, TARGET_CLS]).copy()

    # Discover encoded channel columns created by MultiLabelBinarizer.
    channel_features = [
        col for col in df.columns
        if col.startswith("Channel_") and col != "Channel_Used"
    ]
    feature_columns = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + channel_features
    )
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected features: {missing_features}")

    X = df[feature_columns].copy()
    y_reg = df[TARGET_REG].astype(float)
    y_cls = df[TARGET_CLS].astype(int)

    # Keep the same held-out test rows for both tasks.
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.20, random_state=42, stratify=y_cls
    )
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=16, min_samples_leaf=5, random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=18, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=100, max_depth=18, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )
    }
    cls_models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree Classifier": DecisionTreeClassifier(
            max_depth=16, min_samples_leaf=5, class_weight="balanced", random_state=42
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=18, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Extra Trees Classifier": ExtraTreesClassifier(
            n_estimators=100, max_depth=18, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    }

    reg_results = []
    best_reg_name, best_reg_pipeline, best_r2 = None, None, -np.inf
    print("\n=== REGRESSION: REVENUE ===")
    for name, estimator in reg_models.items():
        pipe = Pipeline([
            ("preprocessor", make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, channel_features)),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_reg_train)
        pred = pipe.predict(X_test)
        mse = mean_squared_error(y_reg_test, pred)
        metrics = {
            "Task": "Regression",
            "Model": name,
            "R2": r2_score(y_reg_test, pred),
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_reg_test, pred),
            "MSE": mse
        }
        reg_results.append(metrics)
        print(f"{name}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
        if metrics["R2"] > best_r2:
            best_r2, best_reg_name, best_reg_pipeline = metrics["R2"], name, pipe

    cls_results = []
    best_cls_name, best_cls_pipeline, best_acc = None, None, -np.inf
    best_cls_predictions = None
    print("\n=== CLASSIFICATION: PROFIT / LOSS ===")
    for name, estimator in cls_models.items():
        pipe = Pipeline([
            ("preprocessor", make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, channel_features)),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_cls_train)
        pred = pipe.predict(X_test)
        metrics = {
            "Task": "Classification",
            "Model": name,
            "Accuracy": accuracy_score(y_cls_test, pred),
            "Precision": precision_score(y_cls_test, pred, zero_division=0),
            "Recall": recall_score(y_cls_test, pred, zero_division=0),
            "F1": f1_score(y_cls_test, pred, zero_division=0)
        }
        cls_results.append(metrics)
        print(
            f"{name}: Accuracy={metrics['Accuracy']:.4f}, "
            f"Precision={metrics['Precision']:.4f}, "
            f"Recall={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}"
        )
        if metrics["Accuracy"] > best_acc:
            best_acc, best_cls_name, best_cls_pipeline = metrics["Accuracy"], name, pipe
            best_cls_predictions = pred

    # Lightweight hyperparameter tuning on a reproducible subset of training data.
    # This is a genuine search; final best model is refitted on all training rows.
    print("\n=== HYPERPARAMETER TUNING ===")
    tune_n = min(25000, len(X_train))
    tune_X = X_train.sample(n=tune_n, random_state=42)
    tune_reg_y = y_reg_train.loc[tune_X.index]
    tune_cls_y = y_cls_train.loc[tune_X.index]

    reg_search_pipe = Pipeline([
        ("preprocessor", make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, channel_features)),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    reg_search = RandomizedSearchCV(
        reg_search_pipe,
        param_distributions={
            "model__n_estimators": [80, 120],
            "model__max_depth": [12, 18, None],
            "model__min_samples_leaf": [2, 4, 6]
        },
        n_iter=4, cv=3, scoring="r2", random_state=42,
        n_jobs=1, verbose=1
    )
    reg_search.fit(tune_X, tune_reg_y)
    tuned_reg = reg_search.best_estimator_
    tuned_reg.fit(X_train, y_reg_train)
    tuned_reg_pred = tuned_reg.predict(X_test)
    tuned_mse = mean_squared_error(y_reg_test, tuned_reg_pred)
    tuned_r2 = r2_score(y_reg_test, tuned_reg_pred)
    reg_results.append({
        "Task": "Regression",
        "Model": "Tuned Random Forest Regressor",
        "R2": tuned_r2,
        "RMSE": np.sqrt(tuned_mse),
        "MAE": mean_absolute_error(y_reg_test, tuned_reg_pred),
        "MSE": tuned_mse
    })
    print(f"Tuned Random Forest Regressor: R2={tuned_r2:.4f}; params={reg_search.best_params_}")
    if tuned_r2 > best_r2:
        best_r2, best_reg_name, best_reg_pipeline = tuned_r2, "Tuned Random Forest Regressor", tuned_reg

    cls_search_pipe = Pipeline([
        ("preprocessor", make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, channel_features)),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))
    ])
    cls_search = RandomizedSearchCV(
        cls_search_pipe,
        param_distributions={
            "model__n_estimators": [80, 120],
            "model__max_depth": [12, 18, None],
            "model__min_samples_leaf": [2, 4, 6]
        },
        n_iter=4, cv=3, scoring="accuracy", random_state=42,
        n_jobs=1, verbose=1
    )
    cls_search.fit(tune_X, tune_cls_y)
    tuned_cls = cls_search.best_estimator_
    tuned_cls.fit(X_train, y_cls_train)
    tuned_cls_pred = tuned_cls.predict(X_test)
    tuned_acc = accuracy_score(y_cls_test, tuned_cls_pred)
    cls_results.append({
        "Task": "Classification",
        "Model": "Tuned Random Forest Classifier",
        "Accuracy": tuned_acc,
        "Precision": precision_score(y_cls_test, tuned_cls_pred, zero_division=0),
        "Recall": recall_score(y_cls_test, tuned_cls_pred, zero_division=0),
        "F1": f1_score(y_cls_test, tuned_cls_pred, zero_division=0)
    })
    print(f"Tuned Random Forest Classifier: Accuracy={tuned_acc:.4f}; params={cls_search.best_params_}")
    if tuned_acc > best_acc:
        best_acc, best_cls_name, best_cls_pipeline = tuned_acc, "Tuned Random Forest Classifier", tuned_cls
        best_cls_predictions = tuned_cls_pred

    # Save all required evaluation metrics.
    evaluation = pd.DataFrame(reg_results + cls_results)
    evaluation.to_csv(REPORT_DIR / "model_evaluation_report.csv", index=False)

    # Save detailed classification report and confusion matrix for the selected model.
    best_cls_predictions = best_cls_pipeline.predict(X_test)
    report = classification_report(y_cls_test, best_cls_predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(REPORT_DIR / "classification_report.csv")
    pd.DataFrame(
        confusion_matrix(y_cls_test, best_cls_predictions),
        index=["Actual_Loss", "Actual_Profit"],
        columns=["Predicted_Loss", "Predicted_Profit"]
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    # Refit chosen models on train partition only; test remains unseen.
    best_reg_pipeline.fit(X_train, y_reg_train)
    best_cls_pipeline.fit(X_train, y_cls_train)

    joblib.dump(best_reg_pipeline, MODEL_DIR / "best_regression_pipeline.joblib")
    joblib.dump(best_cls_pipeline, MODEL_DIR / "best_classification_pipeline.joblib")
    joblib.dump({
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "channel_features": channel_features,
        "all_features": feature_columns
    }, MODEL_DIR / "feature_metadata.joblib")

    metadata = {
        "best_regression_model": best_reg_name,
        "best_classification_model": best_cls_name,
        "regression_test_r2": float(best_r2),
        "classification_test_accuracy": float(best_acc),
        "regression_target": TARGET_REG,
        "classification_target": TARGET_CLS,
        "classification_features_exclude_revenue_roi": True,
        "test_size": 0.20,
        "random_state": 42,
        "regression_benchmark_r2": 0.95,
        "classification_benchmark_accuracy": 0.95,
        "regression_benchmark_met": bool(best_r2 >= 0.95),
        "classification_benchmark_met": bool(best_acc >= 0.95),
        "tuned_regression_params": reg_search.best_params_,
        "tuned_classification_params": cls_search.best_params_
    }
    joblib.dump(metadata, MODEL_DIR / "model_metadata.joblib")
    (REPORT_DIR / "model_summary.json").write_text(json.dumps(metadata, indent=2, default=str))

    print("\n=== SELECTED MODEL RESULTS ON HELD-OUT TEST SET ===")
    print(f"Best regression: {best_reg_name} | R2={best_r2:.4f} | GUVI target 0.95: {'MET' if best_r2 >= 0.95 else 'NOT MET'}")
    print(f"Best classification: {best_cls_name} | Accuracy={best_acc:.4f} | GUVI target 0.95: {'MET' if best_acc >= 0.95 else 'NOT MET'}")
    print(f"Saved artifacts to: {MODEL_DIR}")
    print(f"Elapsed minutes: {(time.time() - start) / 60:.1f}")

if __name__ == "__main__":
    main()
