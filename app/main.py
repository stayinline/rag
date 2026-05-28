from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.weaviate_client import ensure_collection, get_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure Weaviate collection exists
    try:
        client = get_client()
        client.connect()
        try:
            ensure_collection(client)
        finally:
            client.close()
    except Exception as e:
        print(f"Warning: Could not initialize Weaviate: {e}")

    yield

    # Shutdown
    pass


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.v1.router import router as v1_router

app.include_router(v1_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
