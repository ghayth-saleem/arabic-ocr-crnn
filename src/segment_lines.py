"""Classical (non-ML) line segmentation for scanned printed pages.

Uses a horizontal projection profile: sums dark pixels per row, finds
row-bands separated by (mostly) blank rows, and crops each band as one
text line. Works well for clean, single-column printed pages like scanned
book pages; not meant for skewed scans, multi-column layouts, or dense
handwriting.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def binarize(img: Image.Image, threshold: int = 200) -> np.ndarray:
    """Returns a boolean array, True where pixels are ink (dark)."""
    arr = np.array(img.convert("L"), dtype=np.uint8)
    return arr < threshold


def find_line_bands(
    ink_mask: np.ndarray,
    min_ink_ratio: float = 0.01,
    min_gap: int = 4,
    min_line_height: int = 8,
    pad: int = 4,
) -> list[tuple[int, int]]:
    """Returns list of (y_start, y_end) row ranges, one per text line,
    top to bottom."""
    h, w = ink_mask.shape
    row_ink = ink_mask.sum(axis=1) / w
    is_text_row = row_ink > min_ink_ratio

    bands = []
    start = None
    gap = 0
    for y in range(h):
        if is_text_row[y]:
            if start is None:
                start = y
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= min_gap:
                    end = y - gap + 1
                    if end - start >= min_line_height:
                        bands.append((start, end))
                    start = None
                    gap = 0
    if start is not None:
        end = h - gap if gap else h
        if end - start >= min_line_height:
            bands.append((start, end))

    padded = [(max(0, s - pad), min(h, e + pad)) for s, e in bands]
    return padded


def horizontal_extent(ink_mask: np.ndarray, y_start: int, y_end: int, pad: int = 6) -> tuple[int, int]:
    """Tight (x_start, x_end) bounding box of ink within a row band, so
    short lines don't get cropped with huge blank margins (which the
    recognizer, trained on tightly-cropped lines, was never exposed to)."""
    band = ink_mask[y_start:y_end, :]
    col_ink = band.any(axis=0)
    ink_cols = np.where(col_ink)[0]
    if len(ink_cols) == 0:
        return 0, ink_mask.shape[1]
    x_start = max(0, int(ink_cols[0]) - pad)
    x_end = min(ink_mask.shape[1], int(ink_cols[-1]) + 1 + pad)
    return x_start, x_end


def segment_page(image_path: Path, **kwargs) -> list[Image.Image]:
    img = Image.open(image_path).convert("L")
    ink_mask = binarize(img)
    bands = find_line_bands(ink_mask, **kwargs)

    lines = []
    for y_start, y_end in bands:
        x_start, x_end = horizontal_extent(ink_mask, y_start, y_end)
        line_img = img.crop((x_start, y_start, x_end, y_end))
        lines.append(line_img)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = segment_page(Path(args.image))
    print(f"Found {len(lines)} line(s)")
    for i, line_img in enumerate(lines):
        out_path = out_dir / f"line_{i:03d}.png"
        line_img.save(out_path)
        print(f"  {out_path.name}: {line_img.size}")


if __name__ == "__main__":
    main()
