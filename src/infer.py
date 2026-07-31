import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from decode import greedy_decode_texts
from model import CRNN
from vocab import VOCAB_SIZE

TARGET_HEIGHT = 32
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    new_w = max(1, round(img.width * TARGET_HEIGHT / img.height))
    img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def load_model(checkpoint_path: Path, device: torch.device) -> CRNN:
    model = CRNN(VOCAB_SIZE).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def recognize(model: CRNN, image_path: Path, device: torch.device) -> str:
    image = load_image(image_path).unsqueeze(0).to(device)  # (1, 1, H, W)
    log_probs = model(image)  # (T, 1, V)
    seq_len = torch.tensor([CRNN.compute_output_width(image.shape[3])])
    texts = greedy_decode_texts(log_probs.cpu(), seq_len)
    return texts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Image file or directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)

    image_path = Path(args.image)
    if image_path.is_dir():
        paths = sorted(p for p in image_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        paths = [image_path]

    for path in paths:
        text = recognize(model, path, device)
        print(f"{path.name}\t{text}")


if __name__ == "__main__":
    main()
