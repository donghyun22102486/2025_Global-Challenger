from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import tempfile
import os
import joblib
import glob
import json
from datetime import datetime

from llm_handler import create_llm_prompt, query_llm
from preprocess import run_numeric_preprocessing
from EDA import run_basic_eda
from train import train_model

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# 유틸 함수
# -----------------------
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


def run_preprocessing_pipeline(df: pd.DataFrame, user_request: str, save_dir: str):
    # 샘플 & 통계 생성
    n = min(10, len(df))
    sample = pd.concat([df.head(20), df.sample(n=n, random_state=42), df.tail(10)])
    markdown_sample = sample.to_markdown(index=False)
    summary_stats = df.describe(include="all").to_markdown()

    # LLM 호출
    prompt = create_llm_prompt(
        markdown_sample, summary_stats, user_request=user_request
    )
    llm_response, report_text = query_llm(prompt)

    # 전처리 + EDA
    processed_df = run_numeric_preprocessing(df.copy(), llm_response)
    run_basic_eda(processed_df.copy(), save_dir)

    # 저장
    processed_df.to_csv(os.path.join(save_dir, "preprocessed.csv"), index=False)
    with open(os.path.join(save_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    return processed_df, llm_response, report_text


# -----------------------
# API 엔드포인트
# -----------------------
@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    process_option: str = Form(...),  # "full", "preprocess_only", "train_only"
    model_type: str = Form("rf"),  # rf, gbr
    target_col_override: str = Form(None),
    user_request_text: str = Form(None),
    user_request_file: UploadFile = File(None),
):
    try:
        df = read_uploaded_file(file)

        # 사용자 요청 처리
        user_request = ""
        if user_request_text:
            user_request = user_request_text
        elif user_request_file:
            user_request = user_request_file.file.read().decode("utf-8")
        if not user_request:
            user_request = "없음"

        timestamp = datetime.now().strftime("%m%d_%H%M")
        SAVE_DIR = f"results/{timestamp}"
        MODEL_DIR = "models/"
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        results = {}

        if process_option in ["full", "preprocess_only"]:
            processed_df, llm_response, _ = run_preprocessing_pipeline(
                df, user_request, SAVE_DIR
            )

            if process_option == "full":
                target_col = target_col_override or llm_response.get("target_column")
                if not target_col:
                    return {
                        "error": "❌ target_column이 없습니다. override 값이 필요합니다."
                    }
                metrics = train_model(
                    processed_df,
                    target_col=target_col,
                    save_path=MODEL_DIR,
                    model_type=model_type,
                )
                results["metrics"] = metrics

        elif process_option == "train_only":
            target_col = target_col_override or "target"
            metrics = train_model(
                df, target_col=target_col, save_path=MODEL_DIR, model_type=model_type
            )
            results["metrics"] = metrics

        else:
            return {
                "error": "❌ process_option은 full, preprocess_only, train_only 중 하나여야 합니다."
            }

        results.update(
            {
                "timestamp": timestamp,
                "csv_url": f"/download/{timestamp}/csv",
                "report_url": f"/download/{timestamp}/report",
                "eda_url": f"/download/{timestamp}/eda",
            }
        )
        return results

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict-nl")
async def predict_natural_language(
    user_request: str = Form(...), model_file: str = Form(None)
):
    try:
        model_path = get_model_path(model_file)
        model = joblib.load(model_path)
        features = load_model_features(model_path)

        prompt = f"""
        아래는 모델이 예측에 사용하는 feature 목록입니다:
        {features}

        사용자의 요청:
        "{user_request}"

        각 feature에 맞는 값을 JSON 형식으로 채워서 반환하세요.
        없는 값은 0으로 채우고, 숫자는 float로, 문자열은 그대로 두세요.
        """
        llm_response, _ = query_llm(prompt)
        clean_input = {col: llm_response.get(col, 0) for col in features}
        df = pd.DataFrame([clean_input])[features]
        prediction = model.predict(df)

        return {
            "model_used": os.path.basename(model_path),
            "user_request": user_request,
            "parsed_input": clean_input,
            "prediction": float(prediction[0]),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
async def predict(input_data: str = Form(...), model_file: str = Form(None)):
    try:
        input_dict = json.loads(input_data)
        model_path = get_model_path(model_file)
        model = joblib.load(model_path)
        features = load_model_features(model_path)
        clean_input = {col: input_dict.get(col, 0) for col in features}
        df = pd.DataFrame([clean_input])[features]
        prediction = model.predict(df)
        return {
            "model_used": os.path.basename(model_path),
            "input": clean_input,
            "prediction": float(prediction[0]),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/list-models")
async def list_models():
    try:
        model_list = sorted(
            glob.glob("models/*.pkl"), key=os.path.getmtime, reverse=True
        )
        models = [os.path.basename(m) for m in model_list]
        if not models:
            return {"error": "저장된 모델이 없습니다."}
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}


# 공통 유틸
def get_model_path(model_file: str):
    if model_file:
        path = os.path.join("models", model_file)
    else:
        model_list = sorted(
            glob.glob("models/*.pkl"), key=os.path.getmtime, reverse=True
        )
        if not model_list:
            raise FileNotFoundError("저장된 모델이 없습니다.")
        path = model_list[0]
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_file}")
    return path


def load_model_features(model_path: str):
    features_json_path = model_path.replace(".pkl", "_features.json")
    if not os.path.exists(features_json_path):
        raise FileNotFoundError("해당 모델의 feature 목록 파일이 없습니다.")
    with open(features_json_path, "r") as f:
        return json.load(f)


@app.get("/download/{file_id}/{file_type}")
async def download_file(file_id: str, file_type: str):
    file_map = {
        "csv": ("preprocessed.csv", "text/csv"),
        "report": ("report.txt", "text/plain"),
        "eda": ("correlation_heatmap.png", "image/png"),
    }
    if file_type not in file_map:
        return {
            "error": "❌ 지원하지 않는 file_type 입니다. csv, report, eda 중 하나를 사용하세요."
        }
    filename, media_type = file_map[file_type]
    path = os.path.join("results", file_id, filename)
    if not os.path.exists(path):
        return {"error": f"{filename} 파일을 찾을 수 없습니다."}
    return FileResponse(path, filename=filename, media_type=media_type)
