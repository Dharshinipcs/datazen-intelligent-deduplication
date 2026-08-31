from pathlib import Path


# backend/
BACKEND_DIR = Path(__file__).resolve().parents[1]

# Project root/
PROJECT_ROOT = BACKEND_DIR.parent

# Persistent application storage
STORAGE_DIR = BACKEND_DIR / "storage"

RAW_DATA_DIR = STORAGE_DIR / "raw"
PROCESSED_DATA_DIR = STORAGE_DIR / "processed"
METADATA_DIR = STORAGE_DIR / "metadata"
PROFILES_DIR = STORAGE_DIR / "profiles"
RESULTS_DIR = STORAGE_DIR / "results"


# Upload configuration
MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# Application metadata
APP_VERSION = "0.1.0"
