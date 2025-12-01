from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import rag

router = APIRouter()

# --- Request Models (입력 데이터 모델) ---

# 1. 설문조사 요청 (Survey)
class SurveyRequest(BaseModel):
    q1: str
    q2: str
    q3: str
    q4: str
    q5: str
    q6: str
    q7: str
    q8: str
    q9: str
    q10: str

# 2. 키워드 요청 (Keywords)
class KeywordRequest(BaseModel):
    age: str
    gender: str
    relationship: str
    situation: str


# --- Response Models (출력 데이터 모델 - 공통) ---

class GiftItem(BaseModel):
    title: str
    link: str
    image: str
    lprice: str
    mallName: str

class PresentResult(BaseModel):
    analysis: str
    reasoning: str
    card_message: str
    giftList: List[GiftItem]

class ResponseWrapper(BaseModel):
    isSuccess: bool
    code: str
    message: str
    result: Optional[PresentResult] = None

# --- Endpoints (API 주소 정의) ---

# [API 1] 설문조사 결과 추천
# URL: POST /survey/result
@router.post("/survey/result", response_model=ResponseWrapper)
def recommend_by_survey(request: SurveyRequest):
    try:
        # 1. 설문 데이터를 문장으로 변환
        query_sentence = rag.convert_survey_to_query(request.dict())
        
        # 2. RAG 실행
        result_data = rag.get_gift_recommendation(query_sentence)
        
        # 3. 결과 반환
        if not result_data:
             return ResponseWrapper(
                 isSuccess=False, 
                 code="SERVER5001", 
                 message="추천 결과를 생성하지 못했습니다.", 
                 result=None
             )

        return ResponseWrapper(
            isSuccess=True, 
            code="COMMON200", 
            message="성공입니다.", 
            result=PresentResult(**result_data)
        )

    except Exception as e:
        return ResponseWrapper(
            isSuccess=False, 
            code="SERVER5000", 
            message=f"서버 에러: {str(e)}", 
            result=None
        )


# [API 2] 키워드 결과 추천
# URL: POST /keywords/result
@router.post("/keywords/result", response_model=ResponseWrapper)
def recommend_by_keywords(request: KeywordRequest):
    try:
        # 1. 키워드 데이터를 문장으로 변환
        query_sentence = rag.convert_keywords_to_query(request.dict())
        
        # 2. RAG 실행
        result_data = rag.get_gift_recommendation(query_sentence)
        
        # 3. 결과 반환
        if not result_data:
             return ResponseWrapper(
                 isSuccess=False, 
                 code="SERVER5001", 
                 message="추천 결과를 생성하지 못했습니다.", 
                 result=None
             )

        return ResponseWrapper(
            isSuccess=True, 
            code="COMMON200", 
            message="성공입니다.", 
            result=PresentResult(**result_data)
        )

    except Exception as e:
        return ResponseWrapper(
            isSuccess=False, 
            code="SERVER5000", 
            message=f"서버 에러: {str(e)}", 
            result=None
        )

