import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_model(df: pd.DataFrame, target_col: str, save_path: str) -> dict:
    """
    df: 전처리된 DataFrame
    target_col: 예측할 열 이름
    save_path: model.pkl 저장할 디렉토리

    return: 평가지표 dict (rmse, mae, r2)
    """
    if target_col not in df.columns:
        raise ValueError(f"❌ target_col '{target_col}' not in DataFrame")

    X = pd.get_dummies(df.drop(columns=[target_col]))
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    timestamp = datetime.now().strftime("%m%d_%H%M")

    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f"{timestamp}.pkl")
    joblib.dump(model, model_path)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "model_path": model_path,
    }
