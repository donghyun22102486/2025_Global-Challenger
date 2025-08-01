from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import tempfile
import os
from datetime import datetime

from llm_handler import create_llm_prompt, query_llm
from preprocess import run_numeric_preprocessing
from EDA import run_basic_eda
from train import train_model

# FastAPI 앱 생성
app = FastAPI()

# DATA_DIR = "results/data"
# REPORT_DIR = "results/reports"
# EDA_DIR = "results/EDA"
# os.makedirs(DATA_DIR, exist_ok=True)
# os.makedirs(REPORT_DIR, exist_ok=True)


# CORS 설정 (React 등과 연동 시 사용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 파일 읽기 유틸
def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    suffix = file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    if suffix == "csv":
        return pd.read_csv(tmp_path)
    elif suffix in ["xlsx", "xls"]:
        return pd.read_excel(tmp_path)
    else:
        raise ValueError("Only .csv and .xlsx files are supported.")


@app.post("/pipeline")
async def preprocess_file(file: UploadFile = File(...)):
    try:
        ## 전처리 및 보고서 생성 ##

        # 1. 파일 로딩 및 저장 경로 생성
        df = read_uploaded_file(file)
        timestamp = datetime.now().strftime("%m%d_%H%M")

        SAVE_DIR = f"results/{timestamp}"
        MODEL_DIR = "models/"
        os.makedirs(SAVE_DIR, exist_ok=True)

        # 2. 샘플 추출 및 프롬프트 생성
        n = min(10, len(df))
        sample = pd.concat([df.head(20), df.sample(n=n, random_state=42), df.tail(10)])
        markdown_sample = sample.to_markdown(index=False)
        summary_stats = df.describe(include="all").to_markdown()

        prompt = create_llm_prompt(markdown_sample, summary_stats)

        # 3. Gemini 호출 → JSON + 한글 보고서
        llm_response, report_text = query_llm(prompt)

        # 4. 전처리 및 EDA 수행
        processed_df = run_numeric_preprocessing(df.copy(), llm_response)
        run_basic_eda(processed_df.copy(), SAVE_DIR)

        # 5. 파일 저장
        csv_path = os.path.join(SAVE_DIR, f"preprocessed.csv")
        report_path = os.path.join(SAVE_DIR, f"report.txt")

        processed_df.to_csv(csv_path, index=False)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        ## 모델 학습 ##

        target_col = llm_response.get("target_column")

        if not target_col:
            return {"error": "❌ LLM 응답에 target_column 항목이 없습니다."}

        metrics = train_model(processed_df, target_col=target_col, save_path=MODEL_DIR)

        ## 응답 반환 ##
        return {
            "columns": list(processed_df.columns),
            "preview": processed_df.head(5).to_dict(orient="records"),
            "llm_response": llm_response,
            "csv_url": f"/download-csv/{timestamp}",
            "report_url": f"/download-report/{timestamp}",
            "eda_url": f"/download-eda/{timestamp}",
            "model_url": f"/download-model/{timestamp}",
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/download/{file_id}")
async def download_csv(file_id: str):
    path = os.path.join("data", f"{file_id}.csv")
    if not os.path.exists(path):
        return {"error": "CSV 파일을 찾을 수 없습니다."}
    return FileResponse(path, filename=f"{file_id}.csv", media_type="text/csv")


@app.get("/download-report/{file_id}")
async def download_report(file_id: str):
    path = os.path.join("reports", f"{file_id}_report.txt")
    if not os.path.exists(path):
        return {"error": "보고서 파일을 찾을 수 없습니다."}
    return FileResponse(path, filename=f"{file_id}_report.txt", media_type="text/plain")


@app.get("/download-eda/{file_id}")
async def download_eda_heatmap(file_id: str):
    path = os.path.join("results", file_id, "correlation_heatmap.png")
    if not os.path.exists(path):
        return {"error": "EDA 이미지 파일을 찾을 수 없습니다."}
    return FileResponse(
        path, filename="correlation_heatmap.png", media_type="image/png"
    )
