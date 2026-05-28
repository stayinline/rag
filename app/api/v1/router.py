from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.documents import router as documents_router
from app.api.v1.ingestion import router as ingestion_router
from app.api.v1.kbs import router as kbs_router
from app.api.v1.search import router as search_router

router = APIRouter()
router.include_router(kbs_router, prefix="/api/v1")
router.include_router(documents_router, prefix="/api/v1")
router.include_router(ingestion_router, prefix="/api/v1")
router.include_router(chat_router, prefix="/api/v1")
router.include_router(search_router, prefix="/api/v1")
