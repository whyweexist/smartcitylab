"""
Synthetic dataset generation with leakage-safe split.
Supports procedural clean generation + real-world datasets (Picsum, Kodak, DIV2K, MVTec, etc.)
Real-world images are fetched via `backend/ml/fetch_real.py` into data/real_clean.
Hybrid mode merges procedural + real cleans with provenance tagging.

Predefined datasets supported:
- Picsum Photos (CC0, fetched deterministically)
- Kodak PhotoCD (24 images)
- DIV2K / BSDS500 / TID2013 / LIVE / KADID-10k (drop into data/real_clean or data/clean)
- MVTec AD / NEU-CLS (industrial, maps to potential_defect)
"""
import os, json, random, math, shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

ISSUE_TYPES = ["blur","underexposure","overexposure","noise","severe_degradation","potential_defect"]
SEVERITIES = ["low","medium","high"]

def generate_procedural_cleans(out_dir: Path, n=60, size=256, seed=42):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    for i in range(n):
        # Create diverse base: gradient + shapes + texture
        img = np.zeros((size,size,3), dtype=np.uint8)
        # Random gradient
        base_color = rng.randint(40,220,3)
        for y in range(size):
            t = y/size
            col = (base_color * (0.6 + 0.4*t) + rng.randint(-10,10,3)).clip(0,255)
            img[y,:] = col.astype(np.uint8)
        # Add shapes
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        for _ in range(rng.randint(3,8)):
            x0,y0 = rng.randint(0,size,2)
            x1,y1 = rng.randint(0,size,2)
            x0,x1 = sorted([x0,x1]); y0,y1 = sorted([y0,y1])
            color = tuple(rng.randint(0,255,3).tolist())
            shape = rng.choice(["rect","ellipse","line"])
            if shape=="rect":
                draw.rectangle([x0,y0,x1,y1], fill=color, outline=None)
            elif shape=="ellipse":
                draw.ellipse([x0,y0,x1,y1], fill=color)
            else:
                draw.line([x0,y0,x1,y1], fill=color, width=rng.randint(1,5))
        # Add slight texture noise to look natural
        arr = np.array(pil).astype(np.float32)
        arr += rng.randn(size,size,3)*3
        arr = np.clip(arr,0,255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"clean_{i:03d}.jpg", quality=95)

def apply_blur(img, severity, rng):
    k = {"low":3,"medium":7,"high":13}[severity]
    # Gaussian blur + occasional motion
    if rng.rand() < 0.3 and severity!="low":
        # motion blur kernel
        km = np.zeros((k,k))
        km[k//2,:] = 1
        angle = rng.randint(0,180)
        M = cv2.getRotationMatrix2D((k/2,k/2), angle, 1)
        km = cv2.warpAffine(km, M, (k,k))
        km = km / (km.sum()+1e-9)
        return cv2.filter2D(img, -1, km)
    else:
        return cv2.GaussianBlur(img, (k if k%2==1 else k+1, k if k%2==1 else k+1), 0)

def apply_underexposure(img, severity):
    gamma = {"low":1.6,"medium":2.2,"high":3.0}[severity]
    # darken via gamma
    inv = 1.0/gamma
    table = np.array([((i/255.0)**inv)*255 for i in range(256)]).astype("uint8")
    res = cv2.LUT(img, table)
    # also scale brightness
    factor = {"low":0.75,"medium":0.55,"high":0.35}[severity]
    res = (res.astype(np.float32)*factor).clip(0,255).astype(np.uint8)
    return res

def apply_overexposure(img, severity):
    factor = {"low":1.25,"medium":1.55,"high":1.9}[severity]
    res = (img.astype(np.float32)*factor).clip(0,255).astype(np.uint8)
    gamma = {"low":0.85,"medium":0.65,"high":0.45}[severity]
    inv = 1.0/gamma
    table = np.array([((i/255.0)**inv)*255 for i in range(256)]).astype("uint8")
    return cv2.LUT(res, table)

def apply_noise(img, severity, rng):
    sigma = {"low":10,"medium":22,"high":38}[severity]
    noise = rng.randn(*img.shape)*sigma
    if rng.rand() < 0.3:
        # salt & pepper for high
        out = img.astype(np.float32) + noise
        prob = 0.02 if severity=="high" else 0.005
        mask = rng.rand(*img.shape[:2])
        out[mask < prob/2] = 0
        out[mask > 1-prob/2] = 255
        return np.clip(out,0,255).astype(np.uint8)
    return np.clip(img.astype(np.float32)+noise,0,255).astype(np.uint8)

def apply_severe(img, severity, rng):
    # combination heavy: pixelation + compression + heavy blur/noise
    out = img.copy()
    if severity=="low":
        # mild compression artifact simulation: reduce quality
        pass
    # pixelation
    scale = {"low":0.6,"medium":0.35,"high":0.18}[severity]
    h,w = out.shape[:2]
    small = cv2.resize(out, (max(4,int(w*scale)), max(4,int(h*scale))), interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)
    # heavy noise + blur
    out = apply_noise(out, severity, rng)
    out = apply_blur(out, severity, rng)
    # JPEG compression simulation
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), {"low":50,"medium":25,"high":8}[severity]]
    _, enc = cv2.imencode('.jpg', out, encode_param)
    out = cv2.imdecode(enc, 1)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out

def apply_defect(img, severity, rng):
    out = img.copy()
    h,w = out.shape[:2]
    n_defects = {"low":1,"medium":2,"high":4}[severity]
    for _ in range(n_defects):
        x, y = rng.randint(0,w), rng.randint(0,h)
        r = rng.randint(5,18) if severity!="high" else rng.randint(8,30)
        color = tuple(rng.randint(0,255,3).tolist())
        # scratch line or spot
        if rng.rand()<0.5:
            x2,y2 = x + rng.randint(-40,40), y + rng.randint(-40,40)
            cv2.line(out, (x,y), (x2,y2), color, rng.randint(1,4))
        else:
            cv2.circle(out, (x,y), r, color, -1)
            # add ring
            cv2.circle(out, (x,y), r+2, (0,0,0), 1)
    # local corruption block
    if severity=="high" and rng.rand()<0.6:
        x0,y0 = rng.randint(0,w-40), rng.randint(0,h-40)
        out[y0:y0+rng.randint(12,30), x0:x0+rng.randint(12,30)] = rng.randint(0,255,3)
    return out

DEGRAD_FUNCS = {
    "blur": apply_blur,
    "underexposure": apply_underexposure,
    "overexposure": apply_overexposure,
    "noise": apply_noise,
    "severe_degradation": apply_severe,
    "potential_defect": apply_defect,
}

def _infer_source_dataset(cpath: Path) -> str:
    name = cpath.stem.lower()
    # Strip hybrid prefixes proc_/real_ for accurate detection
    raw = name
    if raw.startswith("proc_"):
        raw = raw[5:]
    elif raw.startswith("real_"):
        raw = raw[5:]
    if raw.startswith("picsum"): return "picsum"
    if raw.startswith("kodak") or raw.startswith("kodim"): return "kodak"
    if raw.startswith("clean_"): return "procedural"
    if "mvtec" in raw: return "mvtec"
    if "neu" in raw: return "neu_cls"
    if "div2k" in raw.lower(): return "div2k"
    # check parent folder hint
    if "real_clean" in str(cpath.parent): return "real_world"
    if "hybrid_clean" in str(cpath.parent):
        # fallback by prefix
        if name.startswith("real_"): return "real_world"
        if name.startswith("proc_"): return "procedural"
    if "clean" in str(cpath.parent): return "procedural"
    # hybrid already handled
    if name.startswith("real_"): return "real_world"
    if name.startswith("proc_"): return "procedural"
    return "unknown"

def generate_dataset(clean_dir: str, out_dir: str, manifest_path: str, seed=42, per_clean=6):
    rng = np.random.RandomState(seed)
    random.seed(seed)
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # list cleans — supports nested dirs (for hybrid)
    cleans = sorted([p for p in clean_dir.rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]])
    if not cleans:
        raise ValueError(f"No clean images in {clean_dir}")
    # leakage-safe split: split source ids before generation
    ids = list(range(len(cleans)))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n*0.70)
    n_val = int(n*0.15)
    split_map = {}
    for i, idx in enumerate(ids):
        if i < n_train:
            split_map[idx] = "train"
        elif i < n_train+n_val:
            split_map[idx] = "val"
        else:
            split_map[idx] = "test"
    manifest = []
    for idx, cpath in enumerate(cleans):
        split = split_map[idx]
        # load as RGB
        img_bgr = cv2.imread(str(cpath))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # generate variations: 1 clean + per_clean degradations
        # Clean sample
        fname = f"{cpath.stem}_clean.jpg"
        out_path = out_dir / fname
        # ensure rgb save
        Image.fromarray(img_rgb).save(out_path, quality=95)
        src_ds = _infer_source_dataset(cpath)
        manifest.append({
            "source_id": cpath.stem,
            "image_id": fname,
            "split": split,
            "degradations": [],
            "severity": "none",
            "issues": [],
            "quality_label": "ACCEPTABLE",
            "params": {},
            "source_dataset": src_ds,
            "source_path": str(cpath)
        })
        # degraded samples
        for v in range(per_clean):
            # choose single or multi
            if rng.rand() < 0.35:
                # multi (2 degradations)
                choices = rng.choice(ISSUE_TYPES, size=2, replace=False)
            else:
                choices = [rng.choice(ISSUE_TYPES)]

            severity = rng.choice(SEVERITIES, p=[0.3,0.4,0.3]) if len(choices)==1 else rng.choice(["medium","high"])
            degraded = img_rgb.copy()
            params = {}
            issues = []
            for deg in choices:
                sev = severity if len(choices)==1 else rng.choice(["medium","high"])
                func = DEGRAD_FUNCS[deg]
                if deg in ["blur","noise","severe_degradation","potential_defect"]:
                    degraded = func(degraded, sev, rng)
                else:
                    degraded = func(degraded, sev)
                issues.append(deg)
                params[deg] = sev
            # ensure still RGB
            if degraded.shape[2]==3 and degraded.dtype==np.uint8:
                pass
            else:
                degraded = degraded.astype(np.uint8)
            # handle BGR conversion for severe which returns BGR2RGB already; keep RGB
            if choices[0]=="severe_degradation":
                # already RGB
                pass
            fname2 = f"{cpath.stem}_{'_'.join(choices)}_{severity}_{v}.jpg"
            out_path2 = out_dir / fname2
            # Save: need to handle BGR vs RGB: cv2 vs PIL
            # Use PIL for consistency
            Image.fromarray(degraded).save(out_path2, quality=92)
            # quality label
            if "potential_defect" in issues or "severe_degradation" in issues:
                qlabel = "POTENTIALLY_DEFECTIVE" if rng.rand()<0.7 else "DEGRADED"
            elif len(issues)>=2 or severity=="high":
                qlabel = "DEGRADED"
            else:
                qlabel = "DEGRADED" if rng.rand()<0.8 else "ACCEPTABLE"
            # clean may remain acceptable; degraded noisy etc degrade
            if severity=="high" and len(issues)>=2:
                qlabel = "POTENTIALLY_DEFECTIVE" if "potential_defect" in issues else "DEGRADED"

            manifest.append({
                "source_id": cpath.stem,
                "image_id": fname2,
                "split": split,
                "degradations": issues,
                "severity": severity,
                "issues": issues,
                "quality_label": qlabel,
                "params": params,
                "source_dataset": _infer_source_dataset(cpath),
                "source_path": str(cpath)
            })
    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {len(manifest)} samples, splits: train {sum(1 for m in manifest if m['split']=='train')} val {sum(1 for m in manifest if m['split']=='val')} test {sum(1 for m in manifest if m['split']=='test')}")
    # Provenance summary
    from collections import Counter
    prov = Counter(m.get("source_dataset","unknown") for m in manifest)
    print(f"Source provenance: {dict(prov)}")
    return manifest


def build_hybrid_dataset(procedural_dir="data/clean", real_dir="data/real_clean", out_dir="data/hybrid_clean", manifest_path="data/manifest.json", per_clean=6, seed=42):
    """
    Hybrid real+procedural dataset — recommended production setup.
    Merges procedural cleans (60) + real-world (Picsum/Kodak) into one clean pool,
    then generates leakage-safe variations. Real images improve generalization to
    natural textures, compression artifacts, and sensor noise.
    """
    proc = Path(procedural_dir)
    real = Path(real_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Copy both into hybrid dir with prefixed names to keep source traceable
    for p in sorted(proc.glob("*")):
        if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]:
            shutil.copy2(p, out / f"proc_{p.name}")
    if real.exists():
        for p in sorted(real.rglob("*")):
            if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]:
                # flatten, prefix with real_
                target = out / f"real_{p.name}"
                # handle collisions
                if target.exists():
                    target = out / f"real_{p.stem}_{p.parent.name}{p.suffix}"
                shutil.copy2(p, target)
    print(f"Hybrid clean pool: {len(list(out.glob('*')))} images in {out}")
    return generate_dataset(str(out), "data/generated", manifest_path, seed=seed, per_clean=per_clean)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", default="data/clean")
    ap.add_argument("--out_dir", default="data/generated")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--hybrid", action="store_true", help="Build hybrid procedural+real dataset")
    ap.add_argument("--real_dir", default="data/real_clean")
    ap.add_argument("--hybrid_out", default="data/hybrid_clean")
    ap.add_argument("--per_clean", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.hybrid:
        build_hybrid_dataset(args.clean_dir, args.real_dir, args.hybrid_out, args.manifest, per_clean=args.per_clean, seed=args.seed)
    else:
        generate_dataset(args.clean_dir, args.out_dir, args.manifest, seed=args.seed, per_clean=args.per_clean)
