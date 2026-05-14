from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    return {"status": "ok", "service": "Irozuke AI", "version": "0.1.0"}


@router.get("/health")
def health():
    return {"status": "healthy"}
