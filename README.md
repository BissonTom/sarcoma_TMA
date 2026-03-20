# sarcoma_TMA

`extract_tma_cores.py` is a standalone CLI that:

1. runs TRIDENT tissue segmentation on a directory of whole-slide images,
2. reads the generated `contours_geojson` files,
3. treats each segmented tissue polygon as a TMA core candidate,
4. assigns row/column coordinates by sorting centroids into rows,
5. crops each core from the original slide and saves it as a TIFF named like `slide_r03_c07.tiff`.

The script lives at [scripts/extract_tma_cores.py](/home/tb240/sarcoma_TMA/sarcoma_TMA/scripts/extract_tma_cores.py).

## Requirements

Install the runtime dependencies into the Python environment you will use for extraction:

```bash
pip install numpy tifffile openslide-python
```

You also need:

- a local clone of the TRIDENT repository with its own dependencies installed,
- the OpenSlide shared library available on your system,
- H&E slide files in a format OpenSlide can read.

Run this script from the same Python environment where TRIDENT is installed.

## Example

```bash
python scripts/extract_tma_cores.py \
  --slides-dir /path/to/wsis \
  --output-dir /path/to/tma_core_tiffs \
  --trident-repo /path/to/TRIDENT \
  --segmenter hest \
  --gpu 0 \
  --search-nested
```

If segmentation was already run and `contours_geojson` already exists, reuse it with:

```bash
python scripts/extract_tma_cores.py \
  --slides-dir /path/to/wsis \
  --output-dir /path/to/tma_core_tiffs \
  --trident-repo /path/to/TRIDENT \
  --trident-job-dir /path/to/existing/trident_job \
  --skip-segmentation
```

## Useful Options

- `--padding 64`: expand each crop around the detected core bounding box.
- `--min-core-area`: drop tiny tissue fragments.
- `--max-core-area`: drop abnormally large merged regions.
- `--row-tolerance-factor`: tune row grouping if the grid assignment is off.
- `--overwrite`: replace existing TIFF crops.

## Notes

- The row/column assignment is inferred from polygon centroids, so heavily rotated or irregular TMA layouts may need a larger `--row-tolerance-factor`.
- TRIDENT writes tissue contours to `contours_geojson`; this script uses those contours as the source of truth for cropping.
- Output coordinates are 1-based and zero-padded: `_r01_c01`, `_r01_c02`, and so on.
