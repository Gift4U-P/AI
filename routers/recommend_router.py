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
# URL: POST /survey/result
@router.post("/survey/result")
def recommend_by_survey(request: SurveyRequest):
    try:
        # 1. 설문 데이터를 문장으로 변환
        query_sentence = rag.convert_survey_to_query(request.dict())
        
        # 2. RAG 실행
        result_data = rag.get_survey_recommendation(query_sentence)
        
        # 3. 결과 반환
        if not result_data:
            return {
                "analysis": None,
                "reasoning": None,
                "card_message": None,
                "giftList": []
            }

        # result_data = 분석 + giftList가 이미 포함된 딕셔너리
        return result_data

    except Exception as e:
        return {
            "analysis": None,
            "reasoning": None,
            "card_message": None,
            "giftList": [],
            "error": str(e)
        }

# [API 2] 키워드 결과 추천
# URL: POST /recommend/keywords/result
@router.post("/keywords/result")
def recommend_by_keywords(request: KeywordRequest):
    try:
        # 키워드 전용 함수 호출 (딕셔너리 통째로 전달)
        result_data = rag.get_keyword_recommendation(request.dict())
        
        # 추천 실패 시 빈 값 반환
        if not result_data:
            return {
                "age": None,
                "gender": None,
                "relationship": None,
                "situation": None,
                "keywordText": None,
                "card_message": None,
                "giftList": []
            }

        # result_data는 dict 형태이며, 그대로 반환하면 Spring에서 매핑 가능
        return result_data

    except Exception as e:
        return {
            "age": None,
            "gender": None,
            "relationship": None,
            "situation": None,
            "keywordText": None,
            "card_message": None,
            "giftList": [],
            "error": str(e)
        }