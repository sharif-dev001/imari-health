"""
Financial Health Scoring Engine — XGBoost + SHAP

Uses one-hot encoding for categorical features to avoid the false ordinal
signal that LabelEncoder introduces into tree-based models.

Sources:
    FinScope 2024 (AFR/NISR) — 4-dimension financial health framework, pp. 67-70.
    EICV7 Poverty Profile 2023/24 — poverty-financial gap correlations.
"""
import os, json, joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "../data/processed/training_data.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42

CATEGORICAL = ["gender", "area", "age_group", "education", "income_source",
               "goal_action", "runs_out", "can_raise"]
NUMERIC = ["has_mobile_money", "has_bank_account", "has_savings_account",
           "has_insurance", "has_pension", "has_goal", "controls_finances"]

FEATURES = CATEGORICAL + NUMERIC
TARGET = "segment"
SEGMENT_ORDER = ["extremely_vulnerable", "vulnerable", "coping", "healthy"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def train_model():
    print("=" * 60)
    print("Training XGBoost Financial Health Classifier")
    print("=" * 60)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    X = df[FEATURES]
    y = df[TARGET].map({seg: i for i, seg in enumerate(SEGMENT_ORDER)})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # One-hot encode categoricals — critical fix
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="passthrough",
    )

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.08,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", random_state=RANDOM_STATE,
    )

    pipeline = Pipeline([("prep", preprocessor), ("clf", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=SEGMENT_ORDER, zero_division=0))

    joblib.dump(pipeline, os.path.join(MODEL_DIR, "pipeline.pkl"))
    print(f"\n[OK] Saved pipeline → {MODEL_DIR}/pipeline.pkl")
    return pipeline


def build_explainer(pipeline):
    """Build SHAP explainer on the trained XGBoost stage of the pipeline."""
    clf = pipeline.named_steps["clf"]
    return shap.TreeExplainer(clf)


def get_feature_names(pipeline):
    """Get post-one-hot feature names from the pipeline."""
    prep = pipeline.named_steps["prep"]
    cat_names = prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
    all_names = list(cat_names) + NUMERIC
    return all_names


def predict_one(answers, pipeline, explainer, feature_names):
    """
    Predict + explain. Uses the pipeline for preprocessing so training and
    inference are consistent.
    """
    X = pd.DataFrame([answers])[FEATURES]
    probs = pipeline.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    segment = SEGMENT_ORDER[pred_idx]

    midpoints = np.array([12, 37, 62, 87])
    score = float(np.dot(probs, midpoints))

    # Transform input for SHAP
    X_transformed = pipeline.named_steps["prep"].transform(X)
    sv_list = explainer.shap_values(X_transformed)
    sv = sv_list[pred_idx][0]

    contributions = []
    for i, feat in enumerate(feature_names):
        contributions.append({
            "feature": str(feat),
            "shap": float(sv[i]),
        })
    contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "score": round(score, 1),
        "segment": segment,
        "probabilities": {SEGMENT_ORDER[i]: round(float(probs[i]), 4) for i in range(4)},
        "top_contributions": contributions[:8],
    }


if __name__ == "__main__":
    pipeline = train_model()
    explainer = build_explainer(pipeline)
    feature_names = get_feature_names(pipeline)

    print("\n" + "=" * 60)
    print("Testing prediction on a sample user")
    print("=" * 60)

    # Poor rural female farmer — should be VULNERABLE (~score 25-40)
    sample = {
        "gender": "female", "area": "rural", "age_group": "25-34",
        "education": "primary_or_below", "income_source": "independent_farmer",
        "has_mobile_money": 1, "has_bank_account": 0, "has_savings_account": 0,
        "has_insurance": 1, "has_pension": 0, "has_goal": 1,
        "goal_action": "nothing", "runs_out": "often",
        "can_raise": "no", "controls_finances": 0,
    }
    result = predict_one(sample, pipeline, explainer, feature_names)
    print(json.dumps(result, indent=2))

    # Sanity check: rich urban male — should be HEALTHY
    print("\n" + "=" * 60)
    print("Sanity check: rich urban male")
    print("=" * 60)
    rich = {
        "gender": "male", "area": "urban", "age_group": "35-44",
        "education": "tertiary", "income_source": "non_farm_wage",
        "has_mobile_money": 1, "has_bank_account": 1, "has_savings_account": 1,
        "has_insurance": 1, "has_pension": 1, "has_goal": 1,
        "goal_action": "saving_formally", "runs_out": "rarely",
        "can_raise": "yes", "controls_finances": 1,
    }
    print(json.dumps(predict_one(rich, pipeline, explainer, feature_names), indent=2))