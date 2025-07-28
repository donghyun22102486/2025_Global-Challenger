# llm_handler.py

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일에서 API 키 불러오기
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini 모델 로딩
model = genai.GenerativeModel("gemini-1.5-flash")


def create_llm_prompt(markdown_sample: str, summary_stats: str) -> str:
    return f"""
당신은 다양한 산업 데이터를 다뤄온 10년 경력의 데이터 분석 전문가입니다.  
사용자가 업로드한 데이터 일부와 통계 요약은 아래와 같습니다.

---

**[데이터 샘플 (일부)]**
{markdown_sample}

**[기초 통계 요약]**
{summary_stats}

---

## 🧭 분석 목적

사용자는 이 데이터를 기반으로 후속적인 통계 분석 또는 예측 모델링을 수행할 예정입니다.  
따라서 전처리 단계에서는 컬럼명 표준화, 구조 해석, 파생 변수 정의, 결측치 처리 방안 등이 포함되어야 합니다.

---

## 🛠 요청 작업

아래 ①~④를 순차적으로 수행한 뒤, ⑤에 해당하는 전처리 전문가용 한글 보고서를 작성하세요.

---

### ① 컬럼명 표준화 (`column_mapping`)
- 모든 컬럼명을 간결하고 일관된 영문 소문자 + snake_case 형식으로 표준화하세요.
- 예: `"총 구매금액"` → `"total_purchase_amount"`

### ② 컬럼 그룹화 (`grouping`)
- 관련된 변수들을 하나의 그룹으로 묶고, 그룹의 의미(예: 거래정보, 고객정보)를 간단히 이름 붙여 표현하세요.

### ③ 파생 변수 생성 제안 (`suggested_features`)
- 기존 변수로부터 파생 가능한 유의미한 변수 최대 3개를 제안하세요.
- 다음 구조를 따르세요:

```json
[
  {{
    "name": "sales_excluding_tax",
    "formula": "sales - tax",
    "description": "세금을 제외한 실판매액"
  }},
  ...
]

수식은 pandas df.eval()에서 실행 가능한 형태로 제공하세요.

④ 결측치 처리 전략 (missing_strategy)
데이터셋 전체를 대상으로 가장 적절한 결측치 처리 방식을 하나만 추천하세요.
선택지는: "drop", "mean", "median", "zero", "ffill", "none"

⑤ 전문가용 전처리 보고서 작성
위의 전처리 계획을 바탕으로 다음 구조의 보고서를 작성하세요.
모든 문장은 자연스럽고 정중한 한글 서술형으로 구성하고, 마치 사내 분석 품질팀에 제출하는 공식 문서처럼 작성해주세요.

📄 전처리 전문가 보고서

1. 컬럼 구조 요약 (컬럼명 변경 사유 및 목적)

2. 변수 그룹 구성 및 각 그룹의 의미

3. 파생 변수 설명 (기대 효과 중심)

4. 결측치 처리 전략 설명

5. 전체 전처리 계획 요약

⚠️ 출력 형식

먼저 JSON 응답을 정확한 문법으로 출력하세요.

그 아래에 이어서 한글 보고서 텍스트를 출력하세요.

다른 문장은 포함하지 마세요.

"""


def query_llm(prompt: str):
    try:
        response = model.generate_content(prompt)
        reply = response.text

        # JSON 추출
        start_idx = reply.find("{")
        end_idx = reply.rfind("}") + 1
        json_str = reply[start_idx:end_idx]
        parsed_json = json.loads(json_str)

        # 보고서 텍스트 추출
        report_text = reply[end_idx:].strip()

        return parsed_json, report_text

    except Exception as e:
        raise RuntimeError(f"[Gemini API Error] {e}")
