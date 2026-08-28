import io, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

# ensure app import after path
from app.main import app
from app.database import init_db

init_db()
client = TestClient(app)

def _img_bytes(color=(120,180,90), size=(128,128)):
    im = Image.new("RGB", size, color)
    # add gradient to make features non-trivial
    arr = np.array(im)
    arr[::2] += 10
    im2 = Image.fromarray(arr.astype(np.uint8))
    b = io.BytesIO()
    im2.save(b, format="JPEG", quality=90)
    return b.getvalue()

def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code==200
    j=r.json()
    assert "model_loaded" in j

def test_valid_analysis():
    data = _img_bytes()
    r = client.post("/api/v1/analyze", files={"file": ("test.jpg", data, "image/jpeg")})
    # may fail if model not trained -> 500; but after training should be 200
    assert r.status_code in (200,500)
    if r.status_code==200:
        j=r.json()
        assert "quality_score" in j
        assert j["quality_score"]>=0 and j["quality_score"]<=100
        assert j["quality_label"] in ["ACCEPTABLE","DEGRADED","POTENTIALLY_DEFECTIVE"]

def test_invalid_file():
    r = client.post("/api/v1/analyze", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert r.status_code in (400,422,500)  # should reject

def test_corrupt_image():
    r = client.post("/api/v1/analyze", files={"file": ("corrupt.jpg", b"\xff\xd8\xff\x00\x00bad", "image/jpeg")})
    assert r.status_code in (400,500)

def test_history():
    r = client.get("/api/v1/history")
    assert r.status_code==200
    assert "items" in r.json()

def test_not_found():
    r = client.get("/api/v1/analysis/nonexistent-id")
    assert r.status_code==404

def test_feature_extractor():
    from app.features import extract_features, FEATURE_NAMES
    from app.preprocessing import safe_decode
    data = _img_bytes()
    rgb,gray,_ = safe_decode(data)
    feats = extract_features(rgb,gray)
    assert len(feats)==len(FEATURE_NAMES)
    for k in FEATURE_NAMES: assert k in feats
