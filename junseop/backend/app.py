from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import datetime


# --- 👇 [새로 추가된 Gemini 관련 라이브러리] 👇 ---
import google.generativeai as genai
from dotenv import load_dotenv # python-dotenv 라이브러리 사용
# ---------------------------------------------

# .env 파일에서 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- 모델 및 Gemini API 키 설정 ---
MODEL_PATH = 'fault_prediction_model.pkl'
model_data = {}

# 1. 기존 ML 모델 로드
try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    print(f"'{MODEL_PATH}' 로드 성공.")
except FileNotFoundError:
    print(f"경고: '{MODEL_PATH}'를 찾을 수 없습니다. 'train_model.py'를 먼저 실행해주세요.")

# 2. Gemini API 키 설정
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API 키 설정 완료.")
except Exception as e:
    print(f"Gemini API 키 설정 오류: {e}")
# ------------------------------------

def generate_expert_report(fault_type, features):
    """
    Gemini를 사용하여 Lyra의 고도화된 프롬프트 기반으로 전문가 보고서를 생성합니다.
    """
    # 프롬프트에 넣기 위해 변수들을 가공합니다.
    feature_list_str = ", ".join([f"'{f['feature']}'" for f in features])
    feature_details_str = "\n".join([f"- **{f['feature']} (중요도: {f['importance']:.1%}):**" for f in features])
    current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # --- Lyra님의 프롬프트를 f-string에 맞게 수정한 버전 ---
    # 파이썬 변수가 아닌 placeholder는 {{...}}로 감싸서 SyntaxError를 방지합니다.
    prompt = f"""
# 페르소나 (Persona)
당신은 20년 경력의 철강 제조 공정 품질 관리(QC) 수석 전문가이자, 복잡한 데이터를 현장 작업자가 즉시 행동할 수 있는 명확하고 구체적인 지시로 바꾸는 커뮤니케이션의 대가입니다. 당신의 임무는 단순히 AI의 분석 결과를 전달하는 것이 아니라, 그 안에 숨겨진 의미를 해석하고, 잠재적 위험을 진단하며, 가장 효과적인 해결책을 제시하는 것입니다.

# 컨텍스트 (Context)
방금 AI 실시간 품질 분석 시스템(XGBoost 기반)이 아래와 같은 예측 결과를 보고했습니다. 이 데이터는 기계가 분석한 순수한 정보이며, 당신은 여기에 전문가의 통찰력을 더해야 합니다.
- 예측된 불량 유형: "{fault_type}"
- 불량 발생의 핵심 원인 변수 (Top 5): {feature_list_str}

# 사고 과정 (Chain-of-Thought) - 보고서 작성 전, 다음 단계에 따라 먼저 생각하고 계획을 수립할 것.
1.  **위험 등급 분류:** 예측된 불량 유형('{fault_type}')의 잠재적 심각도와 생산 라인에 미치는 영향을 고려하여, 위험 등급을 [심각], [경고], [주의] 중 하나로 자체적으로 판단한다. (예: 'Z_Scratch'는 제품 외관에 치명적이므로 [심각], 'Stains'는 후공정에서 처리 가능하므로 [주의]로 판단)
2.  **원인 변수 해석:** {feature_list_str} 리스트에 있는 각 변수명이 실제 공정에서 어떤 물리적 의미를 가지는지 구체적으로 연결한다. (예: 'Log_X_Index' -> 강판의 너비 방향 위치, 'TypeOfSteel_A300' -> A300 강재 사용 여부)
3.  **종합적 원인 추론:** 해석된 변수들을 종합하여, 하나의 통합된 문제 상황 시나리오를 논리적으로 추론한다.
4.  **단계별 해결책 설계:** 추론된 문제 상황을 해결하기 위한, 가장 우선순위가 높고 실행 가능한 1, 2, 3단계의 구체적인 행동 계획을 설계한다.

# 최종 보고서 포맷 (Output Format) - 위의 사고 과정을 바탕으로, 아래의 형식을 반드시 준수하여 최종 보고서를 작성할 것.

---

### `[위험 등급 삽입]` **AI 공정 전문가 긴급 리포트**

**1. 요약 보고**
> **{fault_type}** 불량 발생 가능성이 높으며, 핵심 원인은 **{{가장 중요한 변수 1~2개를 이용한 요약}}** 관련 공정으로 보입니다. 즉시 확인이 필요합니다.

**2. 상세 진단**
AI 분석 시스템은 이번 불량 예측의 핵심 원인으로 다음 변수들을 지목했습니다.
{feature_details_str} (각 변수의 물리적 의미와 현재 수치가 왜 문제되는지 상세히 설명)

종합적으로 볼 때, 현재 **{{종합적 원인 추론 결과}}** 상황일 가능성이 매우 높습니다.

**3. [!] 즉시 실행: 단계별 조치 계획**
현장 작업자는 아래 계획에 따라 즉시 점검을 시작해주십시오.

- **1단계 (가장 가능성 높은 원인 확인):**
  - **대상 설비/공정:** {{구체적인 설비명 또는 공정 위치}}
  - **확인 항목:** {{확인해야 할 구체적인 항목, 예: 3번 프레스의 압력 게이지 값}}
  - **정상 기준:** {{정상 상태의 기준값 또는 상태}}

- **2단계 (연관 공정 확인):**
  - **대상 설비/공정:** {{두 번째로 가능성 높은 설비명 또는 공정 위치}}
  - **확인 항목:** {{확인해야 할 구체적인 항목}}
  - **정상 기준:** {{정상 상태의 기준값 또는 상태}}

- **3단계 (원자재/환경 확인):**
  - **대상:** {{투입된 원자재 또는 공정 환경 변수}}
  - **확인 항목:** {{확인해야 할 구체적인 항목}}

**4. 추가 확인 사항**
> 위 3단계 조치로 문제가 해결되지 않을 경우, **{{추가로 점검해볼 만한 사항}}**을 확인하거나 즉시 품질 관리팀에 연락 바랍니다.

**보고 시간:** {current_time_str}
"""
    try:
        llm = genai.GenerativeModel('gemini-1.5-flash')
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API 호출 중 오류 발생: {e}")
        return "전문가 보고서 생성에 실패했습니다. API 키 또는 네트워크 상태를 확인해주세요."


@app.route('/api/predict', methods=['POST'])
def predict_fault():
    # 저장된 모델 데이터가 있는지 확인
    if not model_data:
        return jsonify({"error": "모델이 로드되지 않았습니다."}), 500
    
    xgb_model = model_data.get('model')
    label_encoder = model_data.get('label_encoder')
    feature_names = model_data.get('feature_names')

    try:
        # --- 1단계: 기존 ML 모델로 예측 및 분석 ---
        input_data = request.get_json()
        ordered_input_values = [input_data[name] for name in feature_names]
        input_df = pd.DataFrame([ordered_input_values], columns=feature_names)
        
        prediction_encoded = xgb_model.predict(input_df)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
        
        importances = xgb_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False).head(5)
        top_5_features = feature_importance_df.to_dict(orient='records')
        # ---------------------------------------------

        # --- 2단계: Gemini를 호출하여 전문가 리포트 생성 ---
        expert_report_text = generate_expert_report(prediction_label, top_5_features)
        # ---------------------------------------------

        # --- 3단계: 두 결과를 합쳐서 프론트엔드로 전달 ---
        return jsonify({
            'predicted_fault': prediction_label,
            'root_cause_analysis': top_5_features,
            'expert_report': expert_report_text  # 새로 추가된 키
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)