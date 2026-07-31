"""PyTorch Dataset for synthetic Arabic OCR line images produced by
generate_data.py, plus a collate_fn that right-pads variable-width images
for batching and prepares CTC-ready targets."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vocab import text_to_ctc_indices

TARGET_HEIGHT = 32


class OCRDataset(Dataset):
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.samples: list[tuple[str, str]] = []

        labels_path = self.root_dir / "labels.tsv"
        with open(labels_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                fname, text = line.split("\t", 1)
                self.samples.append((fname, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, text = self.samples[idx]
        img = Image.open(self.images_dir / fname).convert("L")

        if img.height != TARGET_HEIGHT:
            new_w = max(1, round(img.width * TARGET_HEIGHT / img.height))
            img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

        target = torch.tensor(text_to_ctc_indices(text), dtype=torch.long)

        return image_tensor, target, text


def collate_fn(batch):
    images, targets, texts = zip(*batch)

    widths = [img.shape[2] for img in images]
    max_w = max(widths)
    height = images[0].shape[1]

    padded = torch.ones(len(images), 1, height, max_w, dtype=torch.float32)
    for i, img in enumerate(images):
        w = img.shape[2]
        padded[i, :, :, :w] = img

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    concat_targets = torch.cat(targets) if len(targets) > 0 else torch.tensor([], dtype=torch.long)

    input_widths = torch.tensor(widths, dtype=torch.long)

    return padded, concat_targets, target_lengths, input_widths, texts
