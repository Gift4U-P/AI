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
@router.post("/survey/result", response_model=SurveyResult)
def recommend_by_survey(request: SurveyRequest):
    query_sentence, evidence_list, user_trait_keywords = rag.convert_survey_to_query(request.dict())
    result_data = rag.get_survey_recommendation(query_sentence, evidence_list, user_trait_keywords)

    return SurveyResult(**result_data)

@router.post("/keywords/result", response_model=KeywordResult)
def recommend_by_keywords(request: KeywordRequest):
    result_data = rag.get_keyword_recommendation(request.dict())

    return KeywordResult(**result_data)

