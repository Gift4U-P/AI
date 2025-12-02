from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import rag

router = APIRouter()

# --- Request Models ---

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

class KeywordRequest(BaseModel):
    age: str
    gender: str
    relationship: str
    situation: str

# --- Response Models (공통: 선물 아이템) ---

class GiftItem(BaseModel):
    title: str
    link: str
    image: str
    lprice: str
    mallName: str

# --- [설문용] Response Models ---
class SurveyResult(BaseModel):
    analysis: str
    reasoning: str
    card_message: str
    giftList: List[GiftItem]

class SurveyResponseWrapper(BaseModel):
    isSuccess: bool
    code: str
    message: str
    result: Optional[SurveyResult] = None

# --- [키워드용] Response Models (명세서 반영) ---
class KeywordResult(BaseModel):
    age: str
    gender: str
    relationship: str
    situation: str
    keywordText: str
    card_message: str
    giftList: List[GiftItem]

class KeywordResponseWrapper(BaseModel):
    isSuccess: bool
    code: str
    message: str
    result: Optional[KeywordResult] = None


# --- Endpoints ---

# [API 1] 설문조사 결과 추천
# URL: POST /recommend/survey/result
@router.post("/survey/result", response_model=SurveyResponseWrapper)
def recommend_by_survey(request: SurveyRequest):
    try:
        query_sentence = rag.convert_survey_to_query(request.dict())
        
        # 설문 전용 함수 호출
        result_data = rag.get_survey_recommendation(query_sentence)
        
        if not result_data:
             return SurveyResponseWrapper(
                 isSuccess=False, 
                 code="SERVER5001", 
                 message="추천 결과를 생성하지 못했습니다.", 
                 result=None
             )

        return SurveyResponseWrapper(
            isSuccess=True, 
            code="COMMON200", 
            message="성공입니다.", 
            result=SurveyResult(**result_data)
        )

    except Exception as e:
        return SurveyResponseWrapper(
            isSuccess=False, 
            code="SERVER5000", 
            message=f"서버 에러: {str(e)}", 
            result=None
        )


# [API 2] 키워드 결과 추천
# URL: POST /recommend/keywords/result
@router.post("/keywords/result", response_model=KeywordResponseWrapper)
def recommend_by_keywords(request: KeywordRequest):
    try:
        # 키워드 전용 함수 호출 (딕셔너리 통째로 전달)
        result_data = rag.get_keyword_recommendation(request.dict())
        
        if not result_data:
             return KeywordResponseWrapper(
                 isSuccess=False, 
                 code="SERVER5001", 
                 message="추천 결과를 생성하지 못했습니다.", 
                 result=None
             )

        return KeywordResponseWrapper(
            isSuccess=True, 
            code="COMMON200", 
            message="성공입니다.", 
            result=KeywordResult(**result_data)
        )

    except Exception as e:
        return KeywordResponseWrapper(
            isSuccess=False, 
            code="SERVER5000", 
            message=f"서버 에러: {str(e)}", 
            result=None
        )