from fastapi import FastAPI

from app.config import APP_VERSION
from app.routers.profiling import router as profiling_router
from app.routers.upload import router as upload_router
from app.services.storage import initialize_storage


app = FastAPI(
    title="DataZen Intelligent Deduplication",
    description=(
        "Domain-agnostic enterprise entity resolution "
        "and deduplication service."
    ),
    version=APP_VERSION,
)


@app.on_event("startup")
def startup_event() -> None:
    """Initialize application storage when the API starts."""
    initialize_storage()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "datazen-intelligent-deduplication",
        "version": APP_VERSION,
    }


app.include_router(upload_router)
app.include_router(profiling_router)
