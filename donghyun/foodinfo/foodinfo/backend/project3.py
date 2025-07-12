# 실행 uvicorn main:app --reload

from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd

# 데이터 로드
df = pd.read_csv(
    "../../datasets/전국통합식품영양성분정보_음식_표준데이터.csv", encoding="utf-8-sig"
)
df = df.fillna(0)


def classify(source):
    source = str(source)
    if "가정식" in source:
        return "가정식"
    elif "외식" in source or "프랜차이즈" in source:
        return "외식"
    elif "급식" in source:
        return "급식"
    else:
        return "기타"


df["식사형태"] = df["식품기원명"].apply(classify)

# FastAPI 객체 생성
app = FastAPI()

# 선택 가능한 영양 성분
NUTRIENTS = [
    "에너지(kcal)",
    "나트륨(mg)",
    "단백질(g)",
    "지방(g)",
    "탄수화물(g)",
    "식이섬유(g)",
    "칼슘(mg)",
    "철(mg)",
    "비타민 A( g RAE)",
    "콜레스테롤(mg)",
]


# ✅ [GET] 대분류 리스트
@app.get("/categories")
def get_categories():
    return sorted(df["식품대분류명"].unique().tolist())


# ✅ [GET] 영양성분 리스트
@app.get("/nutrients")
def get_nutrients():
    return NUTRIENTS


# ✅ [POST] 분석 요청
class AnalyzeRequest(BaseModel):
    category: str
    nutrients: list


@app.post("/analyze")
def analyze_nutrients(req: AnalyzeRequest):
    category = req.category
    selected = req.nutrients

    filtered_df = df[df["식품대분류명"] == category]
    avg = filtered_df.groupby("식사형태")[selected].mean().reset_index()
    return avg.to_dict(orient="records")


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시엔 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
## error 304 403
