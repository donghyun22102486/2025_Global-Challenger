# pip install google-generativeai

import os
from dotenv import load_dotenv

import google.generativeai as genai


# .env 환경변수 불러오기
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# API 키 설정
genai.configure(api_key=api_key)

# 모델 불러오기
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# 지원하는 모델 확인
# models = genai.list_models()
# for m in models:
#     print(m.name)

# 사용자 프롬프트 입력
prompt = input("Gemini에게 물어보세요: ")

# 답변 받기
response = model.generate_content(prompt)
print(response.text)
