import os
import json
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


def create_llm_prompt(markdown_sample: str, summary_stats: str) -> str:
    with open("prompt_templates/prompt_template.txt", "r", encoding="utf-8") as f:
        raw_template = f.read()

    template = Template(raw_template)
    return template.substitute(
        markdown_sample=markdown_sample, summary_stats=summary_stats
    )


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
