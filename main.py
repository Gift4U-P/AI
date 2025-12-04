from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from routers import recommend_router, users_router
import rag  # 방금 만든 rag.py 모듈 임포트

# 앱 수명 주기 관리 (서버 켜질 때 RAG 초기화)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작될 때 실행
    rag.initialize_rag()
    yield
    # 종료될 때 실행 (필요 시 추가)

app = FastAPI(lifespan=lifespan)

app.include_router(recommend_router.router)
app.include_router(users_router.router)

# 요청 데이터 모델 정의
class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Gift4U API Server is running!"}

