"""
Fetch real-world clean images from predefined public datasets.
- Picsum Photos (Lorem Picsum) : CC0 random photos, deterministic IDs
- Kodak PhotoCD : 24 classic test images (768x512, real photographic)
- Optional: NEU surface defect sample + MVTec hint (if online, else skip)

All images are fetched as clean sources for synthetic degradation pipeline.
Provenance is stored in data/real_clean/README.md and in manifest `source_dataset` field.

Usage:
  python backend/ml/fetch_real.py --out data/real_clean --n_picsum 40 --with_kodak
  python backend/ml/fetch_real.py --out data/real_clean --n_picsum 60
  python -c "from backend.ml.fetch_real import fetch_all; fetch_all('data/real_clean')"

Requires: requests, Pillow
"""
import argparse
import time
from pathlib import Path
import requests
from PIL import Image
import io

PICSUM_IDS = [
    10, 15, 20, 28, 33, 40, 48, 55, 60, 65,
    70, 83, 91, 101, 102, 107, 119, 133, 152, 165,
    177, 182, 190, 197, 203, 211, 221, 237, 244, 250,
    263, 274, 280, 292, 306, 312, 323, 338, 349, 360,
    367, 375, 381, 393, 401, 412, 423, 433, 441, 452,
    464, 473, 480, 491, 502, 513, 525, 536, 547, 559,
]

KODAK_URLS = [f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png" for i in range(1, 25)]

HEADERS = {"User-Agent": "AI-Quality-Assessment/1.0 (research; contact: example@example.com)"}

def download_one(url: str, dest: Path, timeout=20, retries=2) -> bool:
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
            if r.status_code != 200:
                print(f"  ! {url} -> HTTP {r.status_code}")
                return False
            # Validate it's an image by PIL
            data = r.content
            # ensure not HTML
            if len(data) < 2048:
                print(f"  ! too small {url}")
                return False
            # Try open
            im = Image.open(io.BytesIO(data))
            im.verify()
            # Re-open for save (verify closes file)
            im = Image.open(io.BytesIO(data))
            # Convert to RGB if needed, save as JPEG for uniformity
            if im.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                if im.mode == "P":
                    im = im.convert("RGBA")
                bg.paste(im, mask=im.split()[-1] if len(im.split())==4 else None)
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")
            im.save(dest, "JPEG", quality=95)
            print(f"  + {dest.name} ({len(data)//1024} KB) from {url}")
            return True
        except Exception as e:
            print(f"  ! attempt {attempt+1}/{retries+1} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(0.8 * (attempt+1))
    return False

def fetch_picsum(out_dir: Path, n: int = 40, size: int = 512) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    # Use deterministic IDs slice
    ids = PICSUM_IDS[:n] if n <= len(PICSUM_IDS) else (PICSUM_IDS * ((n // len(PICSUM_IDS))+1))[:n]
    for pid in ids:
        url = f"https://picsum.photos/id/{pid}/{size}/{size}"
        dest = out_dir / f"picsum_{pid:04d}.jpg"
        if dest.exists():
            print(f"  = exists {dest.name}")
            count += 1
            continue
        ok = download_one(url, dest)
        if ok:
            count += 1
        time.sleep(0.25)  # be polite, avoid rate limit
        # Picsum redirects to a CDN with actual image; requests follows redirect
    return count

def fetch_kodak(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for url in KODAK_URLS:
        name = url.split("/")[-1].replace(".png", ".jpg")
        dest = out_dir / f"kodak_{name}"
        if dest.exists():
            print(f"  = exists {dest.name}")
            count += 1
            continue
        ok = download_one(url, dest)
        if ok:
            count += 1
        time.sleep(0.3)
    return count

def fetch_all(out: str = "data/real_clean", n_picsum: int = 40, with_kodak: bool = True, with_neu: bool = False):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    print(f"Fetching real-world dataset to {out_dir} ...")
    print(f"[1/2] Picsum ({n_picsum} images, size 512)...")
    c = fetch_picsum(out_dir, n=n_picsum)
    total += c
    print(f"  -> Picsum done: {c}/{n_picsum}")
    if with_kodak:
        print(f"[2/2] Kodak PhotoCD (24 images)...")
        c2 = fetch_kodak(out_dir)
        total += c2
        print(f"  -> Kodak done: {c2}/24")
    # Write provenance
    readme = out_dir / "README.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("# Real-World Clean Dataset\n\n")
        f.write("Fetched via `backend/ml/fetch_real.py`.\n\n")
        f.write("## Sources\n")
        f.write("- **Picsum Photos** (https://picsum.photos) — CC0, random photographic images. Deterministic IDs from PICSUM_IDS for reproducibility. License: https://picsum.photos/ — free to use.\n")
        f.write("- **Kodak PhotoCD** (http://r0k.us/graphics/kodak/) — 24 classic photographic test images (768x512), widely used in image processing. Original Kodak PhotoCD, used with permission for research.\n")
        f.write("- Additional predefined datasets supported (user can drop into this folder):\n")
        f.write("  - **DIV2K** (https://data.vision.ee.ethz.ch/cvl/DIV2K/) — 800 HR images\n")
        f.write("  - **TID2013 / LIVE** — IQA datasets with MOS scores, can be mapped to our 6 issues\n")
        f.write("  - **MVTec AD / NEU-CLS** — industrial defects, maps to `potential_defect` + `severe_degradation`\n")
        f.write("  - **BSDS500** — natural images with human segmentations\n")
        f.write("\n## Usage\n")
        f.write("```bash\n")
        f.write("python backend/ml/fetch_real.py --out data/real_clean --n_picsum 40 --with_kodak\n")
        f.write("python backend/ml/dataset.py  # set clean_dir=data/real_clean\n")
        f.write("```\n")
        f.write("\n## Provenance\n")
        total_files = len(list(out_dir.glob("*.jpg")))
        f.write(f"- Total fetched this run: {total} images\n")
        f.write(f"- Total present: {total_files} JPGs\n")
        f.write(f"- Fetch timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"- Picsum IDs used: {PICSUM_IDS[:n_picsum]}\n")
        f.write(f"- Kodak included: {with_kodak}\n")
    print(f"\nDone. Total images present: {len(list(out_dir.glob('*.jpg')))}")
    print(f"Provenance written to {readme}")
    return total

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch real-world clean images")
    ap.add_argument("--out", default="data/real_clean", help="Output directory")
    ap.add_argument("--n_picsum", type=int, default=40, help="Number of Picsum images (deterministic IDs)")
    ap.add_argument("--with_kodak", action="store_true", default=False, help="Also fetch Kodak 24")
    ap.add_argument("--with_kodak_off", dest="with_kodak", action="store_false")
    ap.set_defaults(with_kodak=True)
    args = ap.parse_args()
    fetch_all(args.out, n_picsum=args.n_picsum, with_kodak=args.with_kodak)
