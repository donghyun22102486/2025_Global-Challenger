# preprocess.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set

# -----------------------------
# 설정(필요시 조정)
# -----------------------------
_MISSING_TOKENS = {"", "-", "--", "na", "n/a", "none", "null", "nan", "무", "없음"}
_DEFAULT_THRESHOLDS = {
    "high_missing_ratio": 0.98,  # 결측 비율 높으면 드랍
    "high_cardinality_ratio": 0.98,  # 문자형 고유값 비율 높으면 ID/키로 판단
    "near_perfect_corr": 0.9999,  # 수치형 상관 ~1이면 중복 판단
}


# ==============================
# 유틸
# ==============================
def _coerce_pseudo_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(
            df[col]
        ):
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({t: np.nan for t in _MISSING_TOKENS}, regex=False)
    return df


def _apply_missing_strategy_series(s: pd.Series, strategy: str) -> pd.Series:
    if strategy == "drop":
        return s.dropna()
    if strategy == "mean" and pd.api.types.is_numeric_dtype(s):
        return s.fillna(s.mean())
    if strategy == "median" and pd.api.types.is_numeric_dtype(s):
        return s.fillna(s.median())
    if strategy == "mode":
        mode = s.mode()
        return s.fillna(mode.iloc[0] if not mode.empty else s)
    if strategy == "zero" and pd.api.types.is_numeric_dtype(s):
        return s.fillna(0)
    if strategy == "ffill":
        return s.fillna(method="ffill")
    if strategy == "none":
        return s
    # fallback
    return (
        s.fillna("missing")
        if not pd.api.types.is_numeric_dtype(s)
        else s.fillna(s.mean())
    )


def _iqr_clip(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s) or s.dropna().empty:
        return s
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return s.clip(lower=lo, upper=hi)


def _zscore_clip(s: pd.Series, z: float = 3.0) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s) or s.dropna().empty:
        return s
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s
    lo, hi = mu - z * sd, mu + z * sd
    return s.clip(lower=lo, upper=hi)


def _apply_outlier_strategy_series(
    s: pd.Series, strategy: str, params: Dict[str, Any] = None
) -> pd.Series:
    params = params or {}
    if strategy == "iqr":
        return _iqr_clip(s)
    if strategy == "zscore":
        return _zscore_clip(s, float(params.get("z", 3.0)))
    if strategy == "clip":
        lo = params.get("min")
        hi = params.get("max")
        if lo is None or hi is None:
            return _iqr_clip(s)  # 안전 클립
        return s.clip(lower=lo, upper=hi)
    return s  # "none" 또는 미지정


# ==============================
# 메인 파이프라인
# ==============================
def run_numeric_preprocessing(df: pd.DataFrame, llm_response: dict) -> pd.DataFrame:
    print("🔧 전처리 시작...")

    # (A) 가짜 결측치 정규화
    df = _coerce_pseudo_missing(df)

    # (A0) 원천 컬럼 정규화: 공백/따옴표 제거 (rename 이전)
    def _strip_quotes_ws(s: str) -> str:
        return str(s).strip().strip("'").strip('"')

    _old = list(df.columns)
    df.columns = [_strip_quotes_ws(c) for c in df.columns]
    if _old != list(df.columns):
        try:
            preview = {
                str(o): str(n) for o, n in zip(_old, df.columns)
            }  # 너무 길면 일부만 출력될 수 있음
            print(
                f"🧼 원천 컬럼 정규화(공백/따옴표 제거) 예시: {list(preview.items())[:10]}"
            )
        except Exception:
            print("🧼 원천 컬럼 정규화 완료")

    # (B) 컬럼명 변경 — LLM 오염 방지(키/값 모두 정제)
    raw_mapping = (llm_response or {}).get("column_mapping", {}) or {}

    def _clean_name(x: str) -> str:
        return str(x).strip().strip("'").strip('"')

    # 1) 키/값 모두 정제
    _cleaned_items = [(_clean_name(k), _clean_name(v)) for k, v in raw_mapping.items()]
    # 2) 현재 DF에 존재하는 키만
    safe_mapping = {k: v for k, v in _cleaned_items if k in df.columns}
    # 3) 값 중복 제거(먼저 등장한 것 우선)
    seen = set()
    dedup_mapping = {}
    for k, v in safe_mapping.items():
        if v not in seen:
            dedup_mapping[k] = v
            seen.add(v)

    if dedup_mapping:
        df.rename(columns=dedup_mapping, inplace=True)
        print(f"✅ 컬럼명 변경(정제) 완료: {list(dedup_mapping.items())[:10]} ...")
    else:
        print("ℹ️ 적용 가능한 컬럼명 변경 없음")

    # (C) 파생변수 생성 — 실패 원인 로깅
    for feature in (llm_response or {}).get("suggested_features", []):
        if not isinstance(feature, dict):
            print(f"⚠️ 잘못된 feature 형식 (dict 아님): {feature}")
            continue
        name = feature.get("name")
        formula = feature.get("formula")
        if not (name and formula):
            print(f"⚠️ name/formula 누락: {feature}")
            continue
        try:
            df[name] = df.eval(formula)
            print(f"✅ {name} 생성 완료 (수식: {formula})")
        except NameError as e:
            print(
                f"⚠️ {name} 생성 실패(NameError): {e} | 사용 가능 컬럼 예시: {list(df.columns)[:10]} ..."
            )
        except Exception as e:
            print(f"⚠️ {name} 생성 실패: {e}")

    # (D) 데이터 정제 추천 + 타깃 보호
    cleaning = (llm_response or {}).get("data_cleaning_recommendations", {}) or {}
    keep_columns: Set[str] = set((cleaning.get("keep_columns") or []))

    target_col = ((llm_response or {}).get("target_column") or "").strip()
    if target_col:
        keep_columns.add(target_col)  # 👈 타깃 강제 보호

    # (E) 임계치 설정
    thr_cfg = (llm_response or {}).get("drop_thresholds", {}) or {}
    high_missing_thr = float(
        thr_cfg.get("high_missing_ratio", _DEFAULT_THRESHOLDS["high_missing_ratio"])
    )
    high_card_thr = float(
        thr_cfg.get(
            "high_cardinality_ratio", _DEFAULT_THRESHOLDS["high_cardinality_ratio"]
        )
    )
    corr_thr = float(
        thr_cfg.get("near_perfect_corr", _DEFAULT_THRESHOLDS["near_perfect_corr"])
    )

    # (F) 힌트 기반 드랍(보호 제외)
    drop_columns_hint = cleaning.get("drop_columns", {}) or {}
    if drop_columns_hint:
        hinted = [
            c
            for c in drop_columns_hint.keys()
            if c in df.columns and c not in keep_columns
        ]
        if hinted:
            df = df.drop(columns=hinted)
            print(f"🗑️ 힌트 기반 드랍: {hinted}")

    # (G) 중복 행 제거
    drop_duplicates_flag = (cleaning.get("drop_duplicates", {}) or {}).get(
        "recommended", False
    )
    if drop_duplicates_flag:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        after = len(df)
        print(f"🧹 중복 행 제거: {before} → {after}")

    # (H1) 상수 컬럼 드랍 — 보호
    drop_cols = [
        c
        for c in df.columns
        if c not in keep_columns and df[c].nunique(dropna=True) <= 1
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"🗑️ 상수 컬럼 드랍: {drop_cols}")

    # (H2) 결측 과다 드랍 — 보호
    miss_ratio = df.isna().mean()
    drop_cols = [
        c
        for c in miss_ratio.index
        if miss_ratio[c] >= high_missing_thr and c not in keep_columns
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"🗑️ 결측 과다 컬럼 드랍(≥{high_missing_thr:.3f}): {drop_cols}")

    # (H3) 고유값 과다 문자 컬럼(ID/키) — 보호
    n = len(df)
    drop_cols = []
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c in keep_columns:
            continue
        nun = df[c].nunique(dropna=True)
        if n > 0 and (nun / n) >= high_card_thr:
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"🗑️ 고유값 과다 문자 컬럼 드랍(≥{high_card_thr:.3f}): {drop_cols}")

    # (H4) 완전 중복 컬럼 드랍 — 보호
    tmp = df.fillna("__NA__SENTINEL__")
    dup_mask = tmp.T.duplicated(keep="first")
    drop_cols = [col for col in tmp.columns[dup_mask] if col not in keep_columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"🗑️ 완전 중복 컬럼 드랍: {drop_cols}")

    # (H5) (수치형) 상관 ~1 중복 컬럼 — 보호
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] > 1:
        corr = num_df.corr()
        cols = num_df.columns.tolist()
        to_drop = set()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                if a in keep_columns or b in keep_columns:
                    continue
                val = corr.iloc[i, j]
                if pd.notna(val) and abs(val) >= corr_thr:
                    to_drop.add(b)  # 뒤에 나온 b를 버림
        drop_cols = sorted(list(to_drop))
        if drop_cols:
            df = df.drop(columns=drop_cols)
            print(f"🗑️ 상관 ~1 중복 컬럼 드랍(≥{corr_thr:.5f}): {drop_cols}")

    # (I) 결측치 처리: 열별 전략 우선 → 전역 전략 보조
    per_col_missing: Dict[str, Dict[str, str]] = (llm_response or {}).get(
        "missing_strategy_per_column", {}
    ) or {}
    global_missing = (llm_response or {}).get("missing_strategy", None)

    if per_col_missing or global_missing:
        for col in list(df.columns):
            if col in per_col_missing:
                strat = per_col_missing[col].get("strategy", "none")
                before_na = df[col].isna().sum()
                df[col] = _apply_missing_strategy_series(df[col], strat)
                after_na = df[col].isna().sum()
                print(f"✅ 결측치 처리({col}): {strat} | {before_na} -> {after_na}")
        if global_missing:
            for col in list(df.columns):
                if df[col].isna().any() and col not in per_col_missing:
                    before_na = df[col].isna().sum()
                    df[col] = _apply_missing_strategy_series(df[col], global_missing)
                    after_na = df[col].isna().sum()
                    print(
                        f"✅ 결측치 처리(전역, {col}): {global_missing} | {before_na} -> {after_na}"
                    )
    else:
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(exclude="number").columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        df[cat_cols] = df[cat_cols].fillna("missing")
        print("✅ 결측치 처리: 기본(mean/missing) 적용")

    # (J) 이상치 처리
    per_col_outlier: Dict[str, Any] = (llm_response or {}).get(
        "outlier_strategy_per_column", {}
    ) or {}
    for col, cfg in per_col_outlier.items():
        if col not in df.columns:
            continue
        strat = cfg.get("strategy", "none")
        params = {k: v for k, v in cfg.items() if k != "strategy"}  # z, min, max 등
        if pd.api.types.is_numeric_dtype(df[col]):
            before_desc = df[col].describe()
            df[col] = _apply_outlier_strategy_series(df[col], strat, params)
            after_desc = df[col].describe()
            print(
                f"✅ 이상치 처리({col}): {strat} | mean {before_desc['mean']:.4f} → {after_desc['mean']:.4f}"
            )
        else:
            print(f"ℹ️ 이상치 처리 스킵({col}): 수치형 아님")

    # (K) 범주형 → 원핫 인코딩
    df = pd.get_dummies(df, drop_first=False)
    print("✅ 범주형 변수 원핫 인코딩 적용 완료")

    print(f"✅ 전처리 완료. 최종 컬럼 수: {len(df.columns)}")
    return df
