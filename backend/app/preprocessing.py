import io
import cv2
import numpy as np
from PIL import Image, ExifTags
from fastapi import HTTPException
from .config import get_settings

settings = get_settings()

def safe_decode(data: bytes):
    try:
        img = Image.open(io.BytesIO(data))
        # EXIF orientation
        try:
            exif = img._getexif()
            if exif is not None:
                orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
                if orientation_key and orientation_key in exif:
                    o = exif[orientation_key]
                    if o == 2:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    elif o == 3:
                        img = img.transpose(Image.ROTATE_180)
                    elif o == 4:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    elif o == 5:
                        img = img.transpose(Image.TRANSPOSE)
                    elif o == 6:
                        img = img.transpose(Image.ROTATE_270)
                    elif o == 7:
                        img = img.transpose(Image.TRANSVERSE)
                    elif o == 8:
                        img = img.transpose(Image.ROTATE_90)
        except Exception:
            pass

        # Convert RGBA/LA/P etc to RGB
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode == "P":
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Invalid dimensions")
        # Protect very large images: resize longest side to max_image_dim
        maxd = settings.max_image_dim
        if max(w, h) > maxd:
            scale = maxd / max(w, h)
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.BICUBIC)
            w, h = nw, nh

        rgb = np.array(img) if img.mode == "RGB" else np.array(img.convert("RGB"))
        # Also provide BGR for cv2? We'll keep RGB and convert where needed
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        return rgb, gray, (w, h)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Corrupt or unreadable image: {e}")

def to_bgr(rgb: np.ndarray):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
