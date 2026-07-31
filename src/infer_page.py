"""Full-page OCR: classical line segmentation (segment_lines.py) followed
by per-line recognition with the trained CRNN (infer.py)."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from decode import greedy_decode_texts
from infer import load_model
from model import CRNN
from segment_lines import segment_page

TARGET_HEIGHT = 32


def line_image_to_tensor(img: Image.Image) -> torch.Tensor:
    new_w = max(1, round(img.width * TARGET_HEIGHT / img.height))
    img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


@torch.no_grad()
def recognize_page(model: CRNN, image_path: Path, device: torch.device, save_lines_dir: Path | None = None):
    lines = segment_page(image_path)
    texts = []
    for i, line_img in enumerate(lines):
        if save_lines_dir is not None:
            save_lines_dir.mkdir(parents=True, exist_ok=True)
            line_img.save(save_lines_dir / f"line_{i:03d}.png")

        tensor = line_image_to_tensor(line_img).unsqueeze(0).to(device)
        log_probs = model(tensor)
        seq_len = torch.tensor([CRNN.compute_output_width(tensor.shape[3])])
        text = greedy_decode_texts(log_probs.cpu(), seq_len)[0]
        texts.append(text)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--save-lines", type=str, default=None, help="Optional dir to dump segmented line crops")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)

    save_dir = Path(args.save_lines) if args.save_lines else None
    texts = recognize_page(model, Path(args.image), device, save_dir)

    for i, text in enumerate(texts):
        print(f"[{i:03d}] {text}")


if __name__ == "__main__":
    main()
