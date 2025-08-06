import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
from string import Template

# .env 파일에서 API 키 불러오기
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini 모델 로딩
model = genai.GenerativeModel("gemini-1.5-flash")


def load_prompt_templates(path="prompt_templates/prompt_template.txt") -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def create_llm_prompt(
    markdown_sample: str, summary_stats: str, user_request: str
) -> str:
    with open("prompt_templates/prompt_template.txt", "r", encoding="utf-8") as f:
        raw_template = f.read()

    template = Template(raw_template)

    return template.substitute(
        markdown_sample=markdown_sample,
        summary_stats=summary_stats,
        user_request=user_request or "없음",
    )


def query_llm(prompt: str):
    try:
        response = model.generate_content(prompt)
        reply = response.text

        # JSON 추출 (정규식 사용)
        json_match = re.search(r"\{.*\}", reply, re.S)
        if not json_match:
            raise ValueError("LLM 응답에서 JSON 객체를 찾을 수 없습니다.")

        json_str = json_match.group(0)

        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 실패: {e}")

        # 보고서 텍스트 추출 (JSON 이후 텍스트)
        report_text = reply[json_match.end() :].strip()

        return parsed_json, report_text

    except Exception as e:
        raise RuntimeError(f"[Gemini API Error] {e}")
