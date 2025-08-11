# Global-Challenger

## 📊 LLM 기반 데이터 전처리 · EDA · 모델 학습 API

이 프로젝트는 **FastAPI**와 **Google Gemini** API를 활용하여  
업로드한 데이터에 대해 **자동 전처리**, **EDA 보고서 생성**, **머신러닝 모델 학습**을 수행하는 백엔드 서비스입니다.

---

## 🚀 주요 기능

- **LLM 기반 전처리 파이프라인**
  - 컬럼명 표준화, 파생변수 생성, 결측치 처리
  - 숫자형 / 범주형 데이터 모두 지원
- **EDA(탐색적 데이터 분석)**
  - 수치형 상관관계 히트맵
  - 결측치 개요, 범주형 변수 빈도 분석
  - 상위 5쌍의 상관관계 분석
- **머신러닝 모델 학습**
  - RandomForest, GradientBoosting 선택 가능
  - Feature importance 저장
  - 5-Fold Cross Validation 결과 제공
- **예측 기능**
  - 일반 JSON 입력 기반 예측
  - 자연어 입력 → LLM 해석 → 예측
- **다운로드 API**
  - 전처리 CSV, 보고서, EDA 히트맵 다운로드

---

## 📂 프로젝트 구조

```bash
project-root/
│
├── app/
│   ├── main.py
│   ├── llm_handler.py
│   ├── preprocess.py
│   ├── train.py
│   ├── EDA.py
│   ├── models/
│   │   └── ...
│   └── results/
│       └── ...
│
├── frontend/
│   └── ...
│
├── data/
│   └── ...
```

## 주요 API

1. /process — 전처리 / EDA / 학습
   POST FormData

file (업로드할 CSV/XLSX)

process_option: "full", "preprocess_only", "train_only"

model_type: "rf" 또는 "gbr"

target_col_override: (옵션) 타겟 컬럼 직접 지정

user_request_text 또는 user_request_file: LLM에 줄 데이터 전처리 지시사항

Response 예시

```json
{
  "timestamp": "0806_1720",
  "csv_url": "/download/0806_1720/csv",
  "report_url": "/download/0806_1720/report",
  "eda_url": "/download/0806_1720/eda",
  "metrics": {
    "rmse": 123.45,
    "mae": 98.76,
    "r2": 0.87,
    "cv_r2": 0.85,
    "model_path": "models/0806_1720.pkl",
    "features_path": "models/0806_1720_features.json",
    "importance_path": "models/0806_1720_feature_importance.json"
  }
}
```

## 2. /predict — JSON 기반 예측

POST FormData

- input_data: 예측할 feature 값 JSON 문자열
- model_file: (옵션) 사용할 모델 파일명

## 3. /predict-nl — 자연어 기반 예측

POST FormData

- user_request: 예측 요청 문장 (자연어)
- model_file: (옵션) 사용할 모델 파일명

LLM이 입력을 해석하여 feature 매핑 후 예측.

## 4. /list-models — 저장된 모델 목록 조회

GET

- 최신 순으로 .pkl 모델 목록 반환

## 5. /download/{file_id}/{file_type} — 결과 파일 다운로드

file_type: "csv", "report", "eda"

예시:

```bash
/download/0806_1720/csv
/download/0806_1720/report
/download/0806_1720/eda
```

## 확장 가능성

- 더 많은 모델 추가 (XGBoost, CatBoost 등)
- UI 개선 (사용자 자유도 확장)
