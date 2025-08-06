import os
import json
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
    """
    전처리된 DataFrame을 학습시켜 모델(.pkl)과 feature 목록(.json), 중요도(.json)을 저장

    Args:
        df (pd.DataFrame): 전처리된 데이터프레임
        target_col (str): 예측할 타겟 컬럼명
        save_path (str): 모델과 feature 파일 저장 폴더 경로
        model_type (str): 사용할 모델 종류 ("rf", "gbr")

    Returns:
        dict: 학습 성능 지표와 저장 경로 정보
    """

    # 1. 타겟 컬럼 검증
    if target_col not in df.columns:
        raise ValueError(
            f"❌ target_col '{target_col}' not found in DataFrame columns."
        )

    # 2. X, y 분리 및 범주형 인코딩
    X = pd.get_dummies(df.drop(columns=[target_col]))
    y = df[target_col]

    # 3. 학습/검증 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. 모델 초기화
    if model_type == "rf":
        model = RandomForestRegressor(random_state=42)
    elif model_type == "gbr":
        model = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("지원하지 않는 모델 타입입니다. 'rf' 또는 'gbr'만 가능합니다.")

    # 5. 모델 학습
    model.fit(X_train, y_train)

    # 6. 예측 및 성능 평가
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # 7. 교차검증 점수 (옵션)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_mean = round(cv_scores.mean(), 4)

    # 8. 저장 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # 9. 모델 저장
    model_path = os.path.join(save_path, f"{timestamp}.pkl")
    joblib.dump(model, model_path)

    # 10. feature 목록 저장
    features_path = os.path.join(save_path, f"{timestamp}_features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, ensure_ascii=False)

    # 11. feature importance 저장 (지원 모델만)
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(X.columns, model.feature_importances_))
        importance_path = os.path.join(
            save_path, f"{timestamp}_feature_importance.json"
        )
        with open(importance_path, "w", encoding="utf-8") as f:
            json.dump(importances, f, ensure_ascii=False)
    else:
        importance_path = None

    # 12. 결과 반환
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "cv_r2": cv_mean,
        "model_path": model_path,
        "features_path": features_path,
        "importance_path": importance_path,
        "timestamp": timestamp,
    }
