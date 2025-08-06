import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_model(
    df: pd.DataFrame, target_col: str, save_path: str, model_type: str = "rf"
) -> dict:
    if target_col not in df.columns:
        raise ValueError(
            f"❌ target_col '{target_col}' not found in DataFrame columns."
        )

    X = pd.get_dummies(df.drop(columns=[target_col]))
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "rf":
        model = RandomForestRegressor(random_state=42)
    elif model_type == "gbr":
        model = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("지원하지 않는 모델 타입입니다. 'rf' 또는 'gbr'만 가능합니다.")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_mean = round(cv_scores.mean(), 4)

    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # 모델 + feature 번들 저장
    model_bundle = {"model": model, "features": list(X.columns)}
    model_path = os.path.join(save_path, f"{timestamp}.pkl")
    joblib.dump(model_bundle, model_path)

    # feature importance 저장
    importance_path = None
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(X.columns, model.feature_importances_))
        importance_path = os.path.join(
            save_path, f"{timestamp}_feature_importance.json"
        )
        with open(importance_path, "w", encoding="utf-8") as f:
            import json

            json.dump(importances, f, ensure_ascii=False)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "cv_r2": cv_mean,
        "model_path": model_path,
        "importance_path": importance_path,
        "timestamp": timestamp,
    }
