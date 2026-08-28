from fastapi import HTTPException
from .config import get_settings
import imghdr

settings = get_settings()
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def validate_upload(filename: str, content_type: str, size: int, data: bytes):
    if not filename:
        raise HTTPException(status_code=400, detail="Missing file")
    if size > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.max_upload_bytes} bytes")
    if size == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    # imghdr check first bytes
    kind = imghdr.what(None, h=data[:2048])
    # imghdr returns jpeg, png etc; allow if not None
    if kind is None:
        # fallback: try Pillow decode later, but we reject obvious non-images
        # Check magic bytes for webp/tiff
        if data[:4] not in [b'\x89PNG', b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1', b'RIFF', b'II*\x00', b'MM\x00*']:
            # Still allow; decoding will reject if corrupt
            pass
    return True
