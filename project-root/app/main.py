# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import tempfile
import os
import joblib
import glob
import json
import traceback
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

from llm_handler import (
    create_llm_prompt,
    query_llm,
    query_llm_plain,
    create_prediction_explanation_prompt,
)
from preprocess import run_numeric_preprocessing
from EDA import run_basic_eda
from train import train_model

# -----------------------------------------------------------------------------
# App / CORS
# -----------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """
    CSV / Excel 업로드 파일을 DataFrame으로 로드.
    """
    suffix = file.filename.split(".")[-1].lower()
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
    """
    LLM 프롬프트 생성 → LLM 응답 파싱 → 전처리 실행 → EDA 산출 → 산출물 저장
    """
    # 프롬프트용 샘플/요약
    n = min(10, len(df))
    sample = pd.concat([df.head(20), df.sample(n=n, random_state=42), df.tail(10)])
    markdown_sample = sample.to_markdown(index=False)
    summary_stats = df.describe(include="all").to_markdown()

    # LLM 호출
    prompt = create_llm_prompt(
        markdown_sample, summary_stats, user_request=user_request
    )
    llm_response, report_text = query_llm(prompt)  # JSON 실패해도 {} + report_text 반환

    # 타깃 보호를 위한 keep_columns 주입(이중 안전망)
    tc = (
        (llm_response.get("target_column") or "").strip()
        if isinstance(llm_response, dict)
        else ""
    )
    if tc:
        dcr = llm_response.get("data_cleaning_recommendations") or {}
        keep = set(dcr.get("keep_columns") or [])
        keep.add(tc)
        dcr["keep_columns"] = list(keep)
        llm_response["data_cleaning_recommendations"] = dcr

    # 전처리/EDA/산출물
    processed_df = run_numeric_preprocessing(df.copy(), llm_response or {})
    run_basic_eda(processed_df.copy(), save_dir)

    processed_path = os.path.join(save_dir, "preprocessed.csv")
    report_path = os.path.join(save_dir, "report.txt")
    processed_df.to_csv(processed_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text or "")

    return processed_df, (llm_response or {}), (report_text or "")


def get_model_bundle(model_file: Optional[str]):
    """
    모델 번들 로드 → (model, features) 반환.
    - 새 포맷: dict(model=..., features=[...])
    - 구 포맷: .pkl + 별도 _features.json
    """
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
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

    bundle = joblib.load(path)

    if isinstance(bundle, dict) and "model" in bundle and "features" in bundle:
        return bundle["model"], bundle["features"]

    # 구버전 호환
    model = bundle
    features_json_path = path.replace(".pkl", "_features.json")
    if not os.path.exists(features_json_path):
        raise FileNotFoundError("해당 모델의 feature 목록 파일이 없습니다.")
    with open(features_json_path, "r", encoding="utf-8") as f:
        features = json.load(f)
    return model, features


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    process_option: str = Form(...),
    model_type: str = Form("rf"),
    target_col_override: Optional[str] = Form(None),
    user_request_text: Optional[str] = Form(None),
    user_request_file: UploadFile = File(None),
):
    """
    process_option: full / preprocess_only / train_only
    """
    try:
        df = read_uploaded_file(file)

        # 사용자 요청 텍스트
        user_request = ""
        if user_request_text:
            user_request = user_request_text
        elif user_request_file:
            user_request = user_request_file.file.read().decode("utf-8")
        if not user_request:
            user_request = "없음"

        timestamp = datetime.now().strftime("%m%d_%H%M")
        SAVE_DIR = f"results/{timestamp}"
        MODEL_DIR = "models"
        _ensure_dirs(SAVE_DIR, MODEL_DIR)

        results: Dict[str, Any] = {}

        if process_option in ["full", "preprocess_only"]:
            processed_df, llm_response, _ = run_preprocessing_pipeline(
                df, user_request, SAVE_DIR
            )

            if process_option == "full":
                # 타깃 결정
                target_col = target_col_override or (
                    llm_response.get("target_column")
                    if isinstance(llm_response, dict)
                    else None
                )
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

        # 산출물 링크
        results.update(
            {
                "timestamp": timestamp,
                "csv_url": f"/download/{timestamp}/csv",
                "report_url": f"/download/{timestamp}/report",
                "eda_url": f"/download/{timestamp}/eda",
                "eda_report_url": f"/download/{timestamp}/eda_report",
            }
        )
        return results

    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


@app.post("/predict-nl")
async def predict_natural_language(
    user_request: str = Form(...), model_file: Optional[str] = Form(None)
):
    """
    자연어 → feature JSON 파싱 → 예측
    """
    try:
        model, features = get_model_bundle(model_file)

        prompt = f"""
아래는 모델이 예측에 사용하는 feature 목록입니다:
{features}

사용자의 요청:
"{user_request}"

각 feature에 맞는 값을 "오직 하나의 JSON"으로만 반환하세요.
키는 feature와 정확히 일치해야 하며, 없는 값은 0으로 두세요.
출력은 반드시 하나의 JSON 코드블록으로 감싸서 주세요.
"""
        llm_response, _ = query_llm(prompt)

        # 최소 유효성 검증
        if not isinstance(llm_response, dict) or not any(
            k in llm_response for k in features
        ):
            return {
                "error": "LLM에서 유효한 입력을 파싱하지 못했습니다. 요청 문장을 더 구체적으로 쓰거나, 숫자값을 포함해 주세요."
            }

        clean_input = {col: llm_response.get(col, 0) for col in features}
        df = pd.DataFrame([clean_input])[features]
        prediction = model.predict(df)

        return {
            "model_used": model_file or "latest",
            "user_request": user_request,
            "parsed_input": clean_input,
            "prediction": float(prediction[0]),
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


@app.post("/predict")
async def predict(input_data: str = Form(...), model_file: Optional[str] = Form(None)):
    """
    JSON 입력 → 예측
    """
    try:
        input_dict = json.loads(input_data)
        model, features = get_model_bundle(model_file)
        clean_input = {col: input_dict.get(col, 0) for col in features}
        df = pd.DataFrame([clean_input])[features]
        prediction = model.predict(df)
        return {
            "model_used": model_file or "latest",
            "input": clean_input,
            "prediction": float(prediction[0]),
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


@app.post("/explain-prediction")
async def explain_prediction(prediction_result: str = Form(...)):
    """
    예측 결과 자연어 설명
    """
    try:
        result_dict = json.loads(prediction_result)
        prompt = create_prediction_explanation_prompt(
            user_request=result_dict["user_request"],
            parsed_input=result_dict["parsed_input"],
            prediction=result_dict["prediction"],
        )
        report = query_llm_plain(prompt)
        return {"explanation": report}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


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
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/download/{file_id}/{file_type}")
async def download_file(file_id: str, file_type: str):
    """
    csv / report / eda(png) / eda_report(txt)
    """
    try:
        file_map = {
            "csv": ("preprocessed.csv", "text/csv"),
            "report": ("report.txt", "text/plain"),
            "eda": ("correlation_heatmap.png", "image/png"),
            "eda_report": ("eda_report.txt", "text/plain"),
        }
        if file_type not in file_map:
            return {
                "error": "❌ 지원하지 않는 file_type 입니다. csv, report, eda, eda_report 중 하나를 사용하세요."
            }

        filename, media_type = file_map[file_type]
        path = os.path.join("results", file_id, filename)
        if not os.path.exists(path):
            return {"error": f"{filename} 파일을 찾을 수 없습니다."}
        return FileResponse(path, filename=filename, media_type=media_type)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}
