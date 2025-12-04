from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import rag

router = APIRouter(prefix="/users")

# --- Response Models ---

class GiftItem(BaseModel):
    title: str
    link: str
    image: str
    lprice: str
    mallName: str

class MainGiftResult(BaseModel):
    randomGifts: List[GiftItem]
    luxuryGifts: List[GiftItem]
    budgetGifts: List[GiftItem]

# --- Endpoint ---

# URL: GET /users/homePresent
@router.get("/homePresent", response_model=MainGiftResult, summary="메인 홈 랜덤 선물 리스트")
def get_main_page_gifts():
    """
    메인 화면에 보여줄 선물 리스트를 반환합니다.
    Wrapper 없이 순수 데이터만 반환합니다.
    """
    try:
        # rag.py 함수 호출
        result_data = rag.get_main_page_gifts()
        
        if result_data is None:
             # 데이터가 준비되지 않았을 때 500 에러 발생
             raise HTTPException(status_code=500, detail="데이터 로드 실패 (서버 초기화 중일 수 있습니다)")

        # [수정] Wrapper 없이 모델 데이터를 직접 반환
        return MainGiftResult(**result_data)

    except Exception as e:
        # 예기치 못한 에러 발생 시 500 에러 발생
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")