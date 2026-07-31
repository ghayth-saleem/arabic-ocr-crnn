# Arabic OCR from scratch (CRNN + CTC)

A CNN + BiLSTM + CTC text-line recognizer for Arabic, trained entirely on
synthetically rendered text (no pretrained weights, no external OCR engine).
Includes a classical line-segmentation step for running the model on full
scanned pages. Sized to train on a 4GB laptop GPU (e.g. RTX 3050).

## How it works

- `generate_data.py` renders random Arabic sentences with random fonts,
  sizes, and light augmentation (rotation/blur/noise) into single-line
  images, producing the synthetic training set.
- `model.py` is a CRNN: a small CNN feature extractor feeding a 2-layer
  BiLSTM, trained with CTC loss so no per-character alignment is needed.
- `segment_lines.py` splits a full page image into individual text lines
  using a horizontal ink-projection profile (classical CV, no ML) so the
  line-level recognizer can be run on real documents.
- `infer_page.py` chains segmentation + recognition to OCR a whole page.

## Setup

```
pip install -r requirements.txt
```

Drop a few Arabic-capable `.ttf`/`.otf` fonts into `fonts/` (not included -
see Licensing below). Free options: [Amiri](https://github.com/aliftype/amiri),
[Noto Naskh Arabic](https://fonts.google.com/noto/specimen/Noto+Naskh+Arabic),
[Cairo](https://fonts.google.com/specimen/Cairo).

## 1. Generate synthetic data

```
cd src
python generate_data.py --out data/synthetic/train --num-samples 200000
python generate_data.py --out data/synthetic/val --num-samples 5000 --seed 999
```

Produces `images/*.png` plus a `labels.tsv` (filename, logical-order Arabic
text) per split.

## 2. Train

```
python train.py --epochs 30 --batch-size 64 --num-workers 4
```

Checkpoints are saved to `checkpoints/last.pt` and `checkpoints/best.pt`
(lowest validation CER seen so far). Resume with:

```
python train.py --resume checkpoints/last.pt --epochs 30
```

## 3. Run inference

Single line image:
```
python infer.py --checkpoint ../checkpoints/best.pt --image path/to/image_or_dir
```

Full page (segmentation + recognition):
```
python infer_page.py --checkpoint ../checkpoints/best.pt --image path/to/page.png
```

## Results

On a held-out synthetic validation set (200k train / 5k val, 30 epochs):
**~4% character error rate (CER)**.

On real scanned book pages (out of the synthetic training distribution -
different font, justified paragraphs, real scan noise): **~20-25% CER**.
The gap is expected - closing it means mixing in real/matching-style
training data rather than relying on synthetic text alone.

## Licensing note

`fonts/` is intentionally excluded from this repo - system fonts (Arial,
Tahoma, Segoe UI, etc.) are proprietary and can't be redistributed. Use
your own licensed or open-source Arabic fonts.

## Next steps

- Mix in real data (e.g. KHATT for handwriting, APTI for printed text) or
  fine-tune on manually transcribed lines from target documents to close
  the synthetic-to-real gap.
- Labels are stored in logical reading order; CTC targets/predictions are
  internally reversed to match the left-to-right image scan direction
  (see `vocab.py`). This assumes lines without embedded Latin/RTL mixing
  beyond isolated digit runs - richer bidi mixing would need proper
  bidi-aware label alignment instead of simple reversal.
