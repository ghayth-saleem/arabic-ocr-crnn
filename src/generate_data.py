"""Generate synthetic Arabic text-line images for OCR training.

Renders random sentences (see corpus.py) with random fonts/sizes onto
images, applies light augmentation, and writes an image + a labels.tsv
file (filename<TAB>logical_text) that dataset.py consumes.
"""

import argparse
import random
from pathlib import Path

import arabic_reshaper
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from corpus import random_sentence

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"
TARGET_HEIGHT = 32


def load_fonts(sizes=(24, 28, 32, 36, 40)) -> list[ImageFont.FreeTypeFont]:
    fonts = []
    for path in sorted(FONTS_DIR.glob("*.ttf")):
        for size in sizes:
            try:
                fonts.append(ImageFont.truetype(str(path), size))
            except OSError:
                continue
    if not fonts:
        raise RuntimeError(f"No usable .ttf fonts found in {FONTS_DIR}")
    return fonts


def to_visual(text: str) -> str:
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def render_line(text: str, font: ImageFont.FreeTypeFont, pad: int = 8) -> Image.Image:
    visual_text = to_visual(text)

    scratch = Image.new("L", (10, 10), color=255)
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), visual_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img_w = text_w + 2 * pad
    img_h = text_h + 2 * pad
    img = Image.new("L", (img_w, img_h), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), visual_text, font=font, fill=0)
    return img


def augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.3:
        angle = random.uniform(-2, 2)
        img = img.rotate(angle, expand=True, fillcolor=255)

    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))

    arr = np.array(img).astype(np.float32)

    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 12), arr.shape)
        arr = arr + noise

    if random.random() < 0.3:
        contrast = random.uniform(0.7, 1.3)
        mean = arr.mean()
        arr = (arr - mean) * contrast + mean

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def resize_to_height(img: Image.Image, target_h: int = TARGET_HEIGHT) -> Image.Image:
    w, h = img.size
    new_w = max(1, round(w * target_h / h))
    return img.resize((new_w, target_h), Image.BILINEAR)


def generate_dataset(out_dir: Path, num_samples: int, seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    fonts = load_fonts()

    labels_path = out_dir / "labels.tsv"
    with open(labels_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            text = random_sentence()
            if not text.strip():
                continue
            font = random.choice(fonts)
            try:
                img = render_line(text, font)
            except Exception:
                continue
            img = augment(img)
            img = resize_to_height(img)

            fname = f"{i:07d}.png"
            img.save(images_dir / fname)
            f.write(f"{fname}\t{text}\n")

            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{num_samples} generated")

    print(f"Done. Wrote {num_samples} samples to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/synthetic/train")
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    generate_dataset(root / args.out, args.num_samples, args.seed)
