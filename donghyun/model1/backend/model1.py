from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import os
import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# --- 환경 설정 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
print("🔑 API KEY 로딩:", GOOGLE_API_KEY)

app = FastAPI()

# --- CORS 허용 설정 (React 개발용 localhost:3000 허용) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 모델 로딩 ---
MODEL_PATH = "../../../models/fault_prediction_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    print("✅ 모델 로드 완료.")
except FileNotFoundError:
    raise RuntimeError("❌ 모델 파일이 존재하지 않습니다.")

xgb_model = model_data["model"]
label_encoder = model_data["label_encoder"]
feature_names = model_data["feature_names"]

print("⚠️ 예측에 필요한 feature 이름들:", feature_names)


# --- Pydantic 모델 정의 ---
from pydantic import BaseModel


class FeatureInput(BaseModel):
    X_Minimum: float
    X_Maximum: float
    Y_Minimum: float
    Y_Maximum: float
    Pixels_Areas: float
    X_Perimeter: float
    Y_Perimeter: float
    Sum_of_Luminosity: float
    Minimum_of_Luminosity: float
    Maximum_of_Luminosity: float
    Length_of_Conveyer: float
    TypeOfSteel_A300: int
    TypeOfSteel_A400: int
    Steel_Plate_Thickness: float
    Edges_Index: float
    Empty_Index: float
    Square_Index: float
    Outside_X_Index: float
    Edges_X_Index: float
    Edges_Y_Index: float
    Outside_Global_Index: float
    LogOfAreas: float
    Log_X_Index: float
    Log_Y_Index: float
    Orientation_Index: float
    Luminosity_Index: float
    SigmoidOfAreas: float


# --- Gemini 전문가 보고서 함수 ---
def generate_expert_report(fault_type, features):
    feature_list_str = ", ".join([f"'{f['feature']}'" for f in features])
    feature_details_str = "\n".join(
        [f"- **{f['feature']} (중요도: {f['importance']:.1%}):**" for f in features]
    )
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = f"""
당신은 20년 경력의 철강 제조 품질 전문가입니다.
- 예측된 불량 유형: "{fault_type}"
- 핵심 원인 변수: {feature_list_str}

### 전문가 보고서 양식:
1. 핵심 요약
2. 변수별 설명
3. 추론된 문제 상황
4. 즉시 실행할 3단계 해결책
5. 추가 확인 사항

{feature_details_str}

보고 시각: {current_time_str}
"""
    try:
        llm = genai.GenerativeModel("gemini-1.5-flash")
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini API 오류:", e)
        return "❌ Gemini 보고서 생성 실패."


# --- 예측 엔드포인트 ---
@app.post("/api/predict")
async def predict_fault(payload: FeatureInput):

    # print("받은 데이터: ", payload)

    input_data = payload.dict()

    # print("✅ feature_names:", feature_names)
    # print("✅ input_data.keys():", input_data.keys())

    try:
        ordered_values = [input_data[feat] for feat in feature_names]
        input_df = pd.DataFrame([ordered_values], columns=feature_names)

        encoded_pred = xgb_model.predict(input_df)
        label = label_encoder.inverse_transform(encoded_pred)[0]

        importances = xgb_model.feature_importances_
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(5)
        )
        top_features = importance_df.to_dict(orient="records")

        report = generate_expert_report(label, top_features)

        return {
            "predicted_fault": label,
            "root_cause_analysis": top_features,
            "expert_report": report,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
