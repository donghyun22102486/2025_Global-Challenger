import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def run_basic_eda(df: pd.DataFrame, save_path: str):
    """
    전처리된 DataFrame을 받아 기본 EDA (히트맵 + 요약 보고서)를 수행하고 파일로 저장한다.
    """
    os.makedirs(save_path, exist_ok=True)

    # 1. 수치형 컬럼만 추출
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    # 2. 히트맵 생성 및 저장
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(save_path, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    # 3. 텍스트 보고서 저장
    report_path = os.path.join(save_path, "eda_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("📊 EDA 요약 보고서\n\n")
        f.write(f"1. 총 데이터 크기: {df.shape[0]} rows, {df.shape[1]} columns\n\n")

        # 결측치 정보
        f.write("2. 결측치 개요:\n")
        nulls = df.isnull().sum()
        for col, cnt in nulls.items():
            if cnt > 0:
                f.write(f"- {col}: {cnt} nulls\n")
        if nulls.sum() == 0:
            f.write("- 없음\n")

        # 주요 상관관계
        f.write("\n3. 상관관계 주요 결과 (절댓값 기준 상위 5쌍):\n")
        top_corr = corr.abs().unstack().sort_values(ascending=False)
        used = set()
        count = 0
        for (a, b), val in top_corr.items():
            if a != b and (b, a) not in used:
                f.write(f"- {a} vs {b}: {corr.loc[a, b]:.2f}\n")
                used.add((a, b))
                count += 1
                if count == 5:
                    break

        f.write("\n👉 전체 상관관계는 heatmap 이미지를 참고하세요.\n")
