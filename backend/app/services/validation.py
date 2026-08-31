from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES


def validate_upload_filename(filename: str | None) -> str:
    """Validate and return the file extension."""
    if not filename or not filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed types: {allowed}",
        )

    return extension


def validate_upload_size(size_bytes: int) -> None:
    """Reject empty files and files exceeding the configured limit."""
    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if size_bytes > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File exceeds the maximum allowed size of "
                f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
            ),
        )


async def save_upload_with_limit(
    upload_file: UploadFile,
    destination: Path,
) -> int:
    """
    Save an uploaded file in chunks while enforcing the size limit.

    Returns:
        Number of bytes written.
    """
    bytes_written = 0
    chunk_size = 1024 * 1024

    try:
        with destination.open("wb") as output_file:
            while chunk := await upload_file.read(chunk_size):
                bytes_written += len(chunk)

                if bytes_written > MAX_UPLOAD_SIZE_BYTES:
                    output_file.close()
                    destination.unlink(missing_ok=True)

                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File exceeds the maximum allowed size of "
                            f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
                        ),
                    )

                output_file.write(chunk)
    finally:
        await upload_file.close()

    validate_upload_size(bytes_written)

    return bytes_written
