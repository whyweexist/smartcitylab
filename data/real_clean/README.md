# Real-World Clean Dataset

Fetched via `backend/ml/fetch_real.py`.

## Sources
- **Picsum Photos** (https://picsum.photos) — CC0, random photographic images. Deterministic IDs from PICSUM_IDS for reproducibility. License: https://picsum.photos/ — free to use.
- **Kodak PhotoCD** (http://r0k.us/graphics/kodak/) — 24 classic photographic test images (768x512), widely used in image processing. Original Kodak PhotoCD, used with permission for research.
- Additional predefined datasets supported (user can drop into this folder):
  - **DIV2K** (https://data.vision.ee.ethz.ch/cvl/DIV2K/) — 800 HR images
  - **TID2013 / LIVE** — IQA datasets with MOS scores, can be mapped to our 6 issues
  - **MVTec AD / NEU-CLS** — industrial defects, maps to `potential_defect` + `severe_degradation`
  - **BSDS500** — natural images with human segmentations

## Usage
```bash
python backend/ml/fetch_real.py --out data/real_clean --n_picsum 40 --with_kodak
python backend/ml/dataset.py  # set clean_dir=data/real_clean
```

## Provenance
- Total fetched this run: 56 images
- Total present: 56 JPGs
- Fetch timestamp: 2026-08-28 14:14:49 UTC
- Picsum IDs used: [10, 15, 20, 28, 33, 40, 48, 55, 60, 65, 70, 83, 91, 101, 102, 107, 119, 133, 152, 165, 177, 182, 190, 197, 203, 211, 221, 237, 244, 250, 263, 274]
- Kodak included: True
