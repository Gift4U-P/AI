from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

import rag

router = APIRouter(tags=["선물 추천 기능"])


# -----------------------------
# Request/Response Models
# -----------------------------
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


class GiftItem(BaseModel):
    title: str
    link: str
    image: str
    lprice: str
    mallName: str
    accuracy: Optional[float] = None


class EvidenceItem(BaseModel):
    category: str
    description: str


class SurveyResult(BaseModel):
    analysis: str
    evidence: List[EvidenceItem]
    reasoning: str
    card_message: str
    giftList: List[GiftItem]


class SurveyResponseWrapper(BaseModel):
    isSuccess: bool
    code: str
    message: str
    result: Optional[SurveyResult] = None


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


# -----------------------------
# APIs
# -----------------------------
@router.post("/survey/result", response_model=SurveyResponseWrapper)
def recommend_by_survey(request: SurveyRequest):
    try:
        query_sentence, evidence_list, user_trait_keywords = rag.convert_survey_to_query(request.dict())
        result_data = rag.get_survey_recommendation(query_sentence, evidence_list, user_trait_keywords)

        if not result_data:
            return SurveyResponseWrapper(isSuccess=False, code="SERVER5001", message="결과 없음", result=None)

        return SurveyResponseWrapper(isSuccess=True, code="COMMON200", message="성공", result=SurveyResult(**result_data))
    except Exception as e:
        return SurveyResponseWrapper(isSuccess=False, code="SERVER5000", message=str(e), result=None)


@router.post("/keywords/result", response_model=KeywordResponseWrapper)
def recommend_by_keywords(request: KeywordRequest):
    try:
        result_data = rag.get_keyword_recommendation(request.dict())

        if not result_data:
            return KeywordResponseWrapper(isSuccess=False, code="SERVER5001", message="결과 없음", result=None)

        return KeywordResponseWrapper(isSuccess=True, code="COMMON200", message="성공", result=KeywordResult(**result_data))
    except Exception as e:
        return KeywordResponseWrapper(isSuccess=False, code="SERVER5000", message=str(e), result=None)
