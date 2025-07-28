# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from llm_handler import create_llm_prompt, query_llm
from preprocess import run_numeric_preprocessing
import pandas as pd
import tempfile
import os
import uuid

# FastAPI 앱 생성
app = FastAPI()
RESULT_DIR = "./results"
os.makedirs(RESULT_DIR, exist_ok=True)

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


@app.post("/preprocess")
async def preprocess_file(file: UploadFile = File(...)):
    try:
        # 1. 파일 로딩
        df = read_uploaded_file(file)

        # 2. 샘플 추출 및 프롬프트 생성
        sample = pd.concat([df.head(20), df.sample(n=10, random_state=42), df.tail(10)])
        markdown_sample = sample.to_markdown(index=False)
        summary_stats = df.describe(include="all").to_markdown()

        prompt = create_llm_prompt(markdown_sample, summary_stats)

        # 3. Gemini 호출 → JSON + 한글 보고서
        llm_response, report_text = query_llm(prompt)

        # 4. 전처리 수행
        processed_df = run_numeric_preprocessing(df.copy(), llm_response)

        # 5. 파일 저장
        file_id = str(uuid.uuid4())
        csv_path = os.path.join(RESULT_DIR, f"{file_id}.csv")
        report_path = os.path.join(RESULT_DIR, f"{file_id}_report.txt")

        processed_df.to_csv(csv_path, index=False)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"✅ 저장 완료: {csv_path}, {report_path}")

        # 6. 응답 반환
        return {
            "columns": list(processed_df.columns),
            "preview": processed_df.head(5).to_dict(orient="records"),
            "llm_response": llm_response,
            "download_url": f"/download/{file_id}",
            "report_url": f"/download-report/{file_id}",
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/download/{file_id}")
async def download_csv(file_id: str):
    path = os.path.join(RESULT_DIR, f"{file_id}.csv")
    if not os.path.exists(path):
        return {"error": "CSV 파일을 찾을 수 없습니다."}
    return FileResponse(
        path, filename=f"processed_{file_id}.csv", media_type="text/csv"
    )


@app.get("/download-report/{file_id}")
async def download_report(file_id: str):
    path = os.path.join(RESULT_DIR, f"{file_id}_report.txt")
    if not os.path.exists(path):
        return {"error": "보고서 파일을 찾을 수 없습니다."}
    return FileResponse(path, filename="expert_report.txt", media_type="text/plain")
