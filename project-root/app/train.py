# train.py
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback


def train_model(
    df: pd.DataFrame, target_col: str, save_path: str, model_type: str = "rf"
) -> Dict:
    """
    - 입력 df에서 target_col을 분리하고 나머지로 원핫 인코딩 후 회귀 모델 학습
    - 타깃이 이진 문자열 등일 경우 숫자로 안전 변환
    - 교차검증 실패해도 저장은 반드시 수행
    - 저장 포맷: joblib.dump({"model": model, "features": [..]})
    """

    def _normalize_name(s: str) -> str:
        s = str(s)
        s = s.strip().strip("'").strip('"')
        s = s.replace(" ", "").replace("_", "").lower()
        return s

    def _resolve_target(col_name: str, columns) -> str:
        norm = _normalize_name(col_name)
        # 1) 완전 일치(정규화 기준)
        for c in columns:
            if _normalize_name(c) == norm:
                return c
        # 2) 복수형/s 보정
        alt = norm + "s" if not norm.endswith("s") else norm[:-1]
        for c in columns:
            if _normalize_name(c) == alt:
                return c
        # 3) 부분 일치(안전하게 길이 4 이상일 때만)
        if len(norm) >= 4:
            for c in columns:
                if norm in _normalize_name(c):
                    return c
        return col_name  # 실패 시 원본 반환

    if target_col not in df.columns:
        raise ValueError(
            f"❌ target_col '{target_col}' not found in DataFrame columns."
        )

    # y 정리: object → 숫자 시도, 실패시 이진 {0,1} 맵핑 시도
    y_raw = df[target_col]
    y = y_raw.copy()

    if y.dtype == "O":
        try:
            y = pd.to_numeric(y, errors="raise")
        except Exception:
            vals = sorted(y_raw.dropna().unique().tolist())
            if len(vals) == 2:
                mapping = {vals[0]: 0, vals[1]: 1}
                y = y_raw.map(mapping)
                print(f"ℹ️ target이 범주형이라 {mapping} 로 매핑했습니다.")
            else:
                raise ValueError(
                    f"❌ target '{target_col}' is non-numeric with {len(vals)} classes: {vals}"
                )

    # X 만들기
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=False)

    # 특성/표본 검증
    if X.shape[0] < 5:
        raise ValueError(f"❌ Too few rows to train: {X.shape[0]}")
    if X.shape[1] == 0:
        raise ValueError("❌ No feature columns after preprocessing (X has 0 columns).")

    # NaN/Inf 처리
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        num_cols = X.select_dtypes(include="number").columns
        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
        obj_cols = X.select_dtypes(exclude="number").columns
        if len(obj_cols):
            X[obj_cols] = X[obj_cols].fillna("missing")
        print("ℹ️ X의 NaN을 기본 전략으로 보정했습니다.")
    if pd.isna(y).any():
        keep_idx = ~pd.isna(y)
        X, y = X.loc[keep_idx], y.loc[keep_idx]
        print(f"ℹ️ y 결측 행 제거: 남은 행 {len(y)}")

    # 학습/평가
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "rf":
        model = RandomForestRegressor(random_state=42)
    elif model_type == "gbr":
        model = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("지원하지 않는 모델 타입입니다. 'rf' 또는 'gbr'만 가능합니다.")

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"❌ 모델 학습 실패: {e}")

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    # 교차검증은 실패해도 저장 계속
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        cv_mean = round(float(cv_scores.mean()), 4)
    except Exception as e:
        print(f"⚠️ cross_val_score 실패(r2): {e}")
        cv_mean = float("nan")

    # 저장
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_bundle = {"model": model, "features": list(X.columns)}
    model_path = os.path.join(save_path, f"{timestamp}.pkl")
    try:
        joblib.dump(model_bundle, model_path)
        print(f"💾 모델 저장 완료: {os.path.abspath(model_path)}")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"❌ 모델 저장 실패: {e}")

    # feature importance 저장(가능할 때만)
    importance_path = None
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(X.columns, model.feature_importances_))
        importance_path = os.path.join(
            save_path, f"{timestamp}_feature_importance.json"
        )
        try:
            import json

            with open(importance_path, "w", encoding="utf-8") as f:
                json.dump(importances, f, ensure_ascii=False)
            print(f"💾 중요도 저장 완료: {os.path.abspath(importance_path)}")
        except Exception as e:
            print(f"⚠️ 중요도 저장 실패: {e}")
            importance_path = None

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "cv_r2": cv_mean,
        "model_path": model_path,
        "importance_path": importance_path,
        "timestamp": timestamp,
    }
