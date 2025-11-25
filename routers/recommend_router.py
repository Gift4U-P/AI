from fastapi import APIRouter

router = APIRouter(prefix="/recommend")

@router.get("/")
def root():
    return {"message": "FastAPI is running!"}


@router.post("/present")
def recommend_present(request: dict):
    return {"result": "present recommendation"}
