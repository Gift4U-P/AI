from fastapi import FastAPI
from routers.recommend_router import router as recommend_router

app = FastAPI()
app.include_router(recommend_router)

