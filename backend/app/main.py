from fastapi import FastAPI

from app.config import APP_VERSION
from app.services.storage import initialize_storage

from app.routers.dedupe import router as dedupe_router
from app.routers.profiling import router as profiling_router
from app.routers.standardization import router as standardization_router
from app.routers.upload import router as upload_router


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
    """Initialize application storage on startup."""
    initialize_storage()


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "datazen-intelligent-deduplication",
        "version": APP_VERSION,
    }


app.include_router(upload_router)
app.include_router(profiling_router)
app.include_router(standardization_router)
app.include_router(dedupe_router)