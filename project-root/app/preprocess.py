import pandas as pd


def run_numeric_preprocessing(df: pd.DataFrame, llm_response: dict) -> pd.DataFrame:
    print("🔧 전처리 시작...")

    # 1. 컬럼명 변경
    column_mapping = llm_response.get("column_mapping", {})
    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)
        print(f"✅ 컬럼명 변경 완료: {column_mapping}")

    # 2. 파생변수 생성 (LLM 수식 기반)
    for feature in llm_response.get("suggested_features", []):
        if isinstance(feature, dict):
            name = feature.get("name")
            formula = feature.get("formula")
            description = feature.get("description", "")
            if name and formula:
                try:
                    df[name] = df.eval(formula)
                    print(f"✅ {name} 생성 완료 (수식: {formula})")
                except Exception as e:
                    print(f"⚠️ {name} 생성 실패: {e}")
            else:
                print(f"⚠️ name 또는 formula 누락: {feature}")
        else:
            print(f"⚠️ 잘못된 feature 형식 (dict 아님): {feature}")

    # 3. 결측치 처리
    strategy = llm_response.get("missing_strategy", "mean")
    try:
        if strategy == "drop":
            df.dropna(inplace=True)
        elif strategy == "mean":
            df.fillna(df.mean(numeric_only=True), inplace=True)
        elif strategy == "median":
            df.fillna(df.median(numeric_only=True), inplace=True)
        elif strategy == "zero":
            df.fillna(0, inplace=True)
        elif strategy == "ffill":
            df.fillna(method="ffill", inplace=True)
        elif strategy == "none":
            pass
        else:
            print(f"⚠️ 알 수 없는 결측치 처리 방식: {strategy}")
        print(f"✅ 결측치 처리 방식 적용: {strategy}")
    except Exception as e:
        print(f"⚠️ 결측치 처리 실패: {e}")

    # 4. 범주형 → 원핫 인코딩
    df = pd.get_dummies(df)
    print("✅ 범주형 변수 원핫 인코딩 적용 완료")

    print(f"✅ 전처리 완료. 최종 컬럼: {df.columns.tolist()}")
    return df
