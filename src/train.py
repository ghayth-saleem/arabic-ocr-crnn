import argparse
from pathlib import Path

import editdistance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OCRDataset, collate_fn
from decode import greedy_decode_texts
from model import CRNN
from vocab import BLANK_IDX, VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent


def compute_cer(preds: list[str], targets: list[str]) -> float:
    total_dist = 0
    total_len = 0
    for pred, target in zip(preds, targets):
        total_dist += editdistance.eval(pred, target)
        total_len += max(1, len(target))
    return total_dist / total_len


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for images, _, _, input_widths, texts in loader:
        images = images.to(device)
        log_probs = model(images)
        seq_lens = torch.clamp(input_widths // 4 - 1, min=1)
        preds = greedy_decode_texts(log_probs.cpu(), seq_lens)
        all_preds.extend(preds)
        all_targets.extend(texts)
    model.train()
    return compute_cer(all_preds, all_targets), list(zip(all_preds[:5], all_targets[:5]))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = OCRDataset(ROOT / args.train_dir)
    val_ds = OCRDataset(ROOT / args.val_dir)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    model = CRNN(VOCAB_SIZE).to(device)

    start_epoch = 0
    best_cer = float("inf")
    ckpt_dir = ROOT / args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if args.resume:
        resume_path = ROOT / args.resume
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_cer = ckpt.get("best_cer", float("inf"))
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        running_loss = 0.0

        for step, (images, targets, target_lengths, input_widths, texts) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            input_lengths = torch.clamp(input_widths // 4 - 1, min=1).to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                log_probs = model(images)  # (T, B, V)
                loss = criterion(log_probs, targets, input_lengths, target_lengths.to(device))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if (step + 1) % 20 == 0:
                pbar.set_postfix(loss=running_loss / (step + 1))

        avg_loss = running_loss / len(train_loader)
        cer, samples = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={avg_loss:.4f} val_CER={cer:.4f}")
        for pred, target in samples:
            print(f"  pred='{pred}'  target='{target}'")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_cer": min(best_cer, cer),
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if cer < best_cer:
            best_cer = cer
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  New best CER: {best_cer:.4f} -> saved best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="data/synthetic/train")
    parser.add_argument("--val-dir", type=str, default="data/synthetic/val")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train(args)
