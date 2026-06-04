from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router
from app.config import settings
from app.logging_config import RequestLoggingMiddleware, setup_logging
from app.services.weaviate_client import close_client, ensure_collection, get_client

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure Weaviate collection exists
    logger.info("Application startup begin app_name=%s debug=%s", settings.app_name, settings.debug)
    try:
        logger.info(
            "Initializing Weaviate collection url=%s grpc_port=%s",
            settings.weaviate_url,
            settings.weaviate_grpc_port,
        )
        client = get_client()
        ensure_collection(client)
        logger.info("Weaviate collection ready")
    except Exception as e:
        logger.warning("Could not initialize Weaviate during startup: %s", e, exc_info=True)

    yield

    # Shutdown
    close_client()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
