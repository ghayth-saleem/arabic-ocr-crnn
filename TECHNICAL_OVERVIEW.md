# Arabic OCR From Scratch — Full Technical Walkthrough

This document explains the project end to end: every technique used, the
competing techniques that were *not* used, how each one actually works
under the hood, and why the choice was made. Written in English on purpose
— mixing Arabic RTL text with inline English technical terms (CNN, CTC,
BiLSTM...) renders badly in some terminal/chat UIs.

---

## Table of contents

1. [The problem, restated](#1-the-problem-restated)
2. [The big architectural choice: segmentation-free recognition](#2-the-big-architectural-choice-segmentation-free-recognition)
3. [The CNN backbone](#3-the-cnn-backbone)
4. [The sequence model: BiLSTM](#4-the-sequence-model-bilstm)
5. [CTC loss — the deepest part of this project](#5-ctc-loss--the-deepest-part-of-this-project)
6. [Decoding: turning probabilities into text](#6-decoding-turning-probabilities-into-text)
7. [Data: synthetic generation vs. alternatives](#7-data-synthetic-generation-vs-alternatives)
8. [Arabic text shaping](#8-arabic-text-shaping)
9. [Vocabulary design](#9-vocabulary-design)
10. [Training infrastructure](#10-training-infrastructure)
11. [Page-level OCR: line segmentation](#11-page-level-ocr-line-segmentation)
12. [The synthetic-to-real gap](#12-the-synthetic-to-real-gap)
13. [File-by-file code map](#13-file-by-file-code-map)
14. [Honest limitations](#14-honest-limitations)

---

## 1. The problem, restated

OCR (Optical Character Recognition) means: given an image containing text,
output the text as a string. That sounds like classification, but it
isn't a normal classification problem, for one core reason:

**We don't know which pixels correspond to which character.**

A training example is just `(image, "hello")` — nobody drew bounding
boxes around each letter. This single fact is what drives almost every
architectural decision below.

---

## 2. The big architectural choice: segmentation-free recognition

### What we used: CRNN (CNN + RNN + CTC)

The pipeline treats a text line as a *sequence*: scan the image left to
right, and at every point in the scan, guess what character (if any) is
being written there. A CNN extracts visual features, an RNN adds context,
and a loss function called CTC (section 5) handles the fact that we don't
know the true pixel-to-character alignment.

### Competing approach #1: Character segmentation + classifier (the "classical" pipeline)

Older OCR systems (and how humans might first think to solve this):
1. Find the boundaries between characters (segmentation) using pixel
   projections, connected components, or contour analysis.
2. Crop each character out.
3. Run a simple classifier (even a basic CNN, or historically a
   Support Vector Machine on handcrafted features) on each crop.

**Why we didn't use this**: it completely breaks for Arabic. Arabic
letters are *connected* (cursive by default) and change shape by
position (isolated/initial/medial/final). There often isn't a clean
vertical boundary between two letters to cut at — the whole point of
CTC-based recognition is to avoid ever having to answer "where does this
letter end and the next begin." Segmentation-based OCR works reasonably
for isolated-print Latin text; it's a poor fit for Arabic.

### Competing approach #2: Attention-based encoder-decoder (seq2seq)

Instead of CTC, encode the image into a feature sequence, then use a
decoder RNN (or Transformer) that, at each output step, uses an
**attention mechanism** to look back at the relevant part of the image
and predict the next character — very similar to how machine translation
models work (image → source language, text → target language).

**How attention works, briefly**: at each decoding step, the decoder
computes a similarity score between its current hidden state and every
position in the encoded image sequence, turns those scores into weights
via softmax, and takes a weighted sum of the image features as "the part
of the image I should look at right now." This soft-alignment is learned
end to end.

**Why we didn't use this**: attention-based decoders are harder to train
from scratch on a small dataset/small compute budget — they tend to need
more data to learn stable alignments, are slower at inference (decoding
is autoregressive, one character at a time, versus CTC's one parallel
forward pass), and are more complex to implement correctly (attention
masking, teacher forcing during training, exposure bias). CTC is simpler,
faster, and works well specifically for cases (like OCR) where input and
output are roughly monotonically aligned — the image scan order and
character order never "jump around," which is exactly the condition CTC
is built for. Attention shines when alignment is *not* monotonic (e.g.
translation, where word order changes between languages); OCR doesn't
need that flexibility.

### Competing approach #3: Full Transformer, end-to-end (TrOCR-style)

Vision Transformer (ViT) encoder + Transformer text decoder, both
pretrained on massive datasets, then fine-tuned for OCR. This is the
current state of the art in general-purpose OCR (used by tools like
Microsoft's TrOCR, and how modern multimodal models like GPT-4V/Gemini
read images).

**Why we didn't use this**: these models are enormous (tens to hundreds
of millions of parameters minimum, often much more) and their strong
results depend heavily on large-scale pretraining. Training one *from
scratch* on a single 4GB laptop GPU with a synthetic dataset is not
realistic — you'd need a much bigger compute budget and far more data
than what generates in a few minutes locally. CRNN+CTC is the right size
for this hardware and this "from scratch" constraint.

### Competing approach #4: RNN-Transducer (RNN-T)

Popular in speech recognition (this is what a lot of streaming speech
recognizers use). Combines an encoder (like our CNN+BiLSTM), a
prediction network (a small language model over previously predicted
characters), and a joint network that combines both to predict the next
symbol — allowing the model to condition on its own output history,
unlike CTC which assumes each output step is conditionally independent
given the input.

**Why we didn't use this**: more moving parts, more complex to implement
and train stably, and its main advantage (better implicit language
modeling, useful for speech where acoustic ambiguity is high) matters
less for printed-text OCR where the visual signal is much cleaner. CTC's
simplicity was the better trade for a from-scratch project.

### Summary table

| Approach | Alignment strategy | Training complexity | Inference speed | Data/compute needed | Verdict |
|---|---|---|---|---|---|
| **CTC (used)** | Learned implicitly, monotonic | Low | Fast (parallel) | Moderate | ✅ chosen |
| Character segmentation | Manual/heuristic | Low (per-char) | Fast | Low, but fragile for cursive script | ❌ breaks on Arabic |
| Attention seq2seq | Learned, non-monotonic | Medium-high | Slow (autoregressive) | High | ❌ overkill here |
| Transformer end-to-end | Learned, non-monotonic | High | Slow | Very high | ❌ needs pretraining at scale |
| RNN-Transducer | Learned, monotonic + autoregressive | High | Medium | High | ❌ unnecessary complexity |

---

## 3. The CNN backbone

From [model.py](src/model.py):

```python
self.cnn = nn.Sequential(
    nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
    nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
    nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
    nn.MaxPool2d((2, 1), (2, 1)),
    nn.Conv2d(512, 512, 2, padding=0), nn.BatchNorm2d(512), nn.ReLU(),
)
```

This is a small **VGG-style** CNN (stacked 3x3 convolutions + max pooling,
the same design family as the 2014 VGGNet). Notice the pooling layers
switch from `(2,2)` to `(2,1)` in the later layers — this shrinks the
*height* faster than the *width*. That's deliberate: we want to end up
with height = 1 (one feature vector per horizontal position) while
keeping enough width resolution to distinguish individual characters.
This exact backbone is the one from the original 2015 CRNN paper
(Shi, Bai, Yao — "An End-to-End Trainable Neural Network for Image-based
Sequence Recognition").

### Alternatives considered

| Backbone | Idea | Why not used |
|---|---|---|
| **ResNet** | Adds skip/residual connections so gradients flow through very deep networks without vanishing | Solves a problem (training very deep nets) we don't have — our images are small (32px tall) and don't need extreme depth. Adds parameters/compute for no real benefit here. |
| **EfficientNet** | Systematically scales depth/width/resolution together for best accuracy-per-FLOP | Designed and tuned for large-scale image classification (ImageNet); overkill and non-trivial to adapt to a 1-channel, height-32 recognition task |
| **MobileNet** (depthwise-separable convs) | Fewer parameters/FLOPs, meant for phones | Would help if we were compute-starved on *inference*, but our bottleneck was training data/time, not inference speed; plain convs are simpler to reason about |
| **Vision Transformer patches** | Split image into patches, treat as a token sequence, self-attention | Needs much more data to train from scratch (ViTs famously underperform CNNs without either huge datasets or pretraining) — wrong fit for a small synthetic dataset |

The VGG-style CNN was picked because it's the smallest, simplest thing
that reliably works for this exact problem — it's the architecture the
original CRNN paper validated, and it fits comfortably in 4GB of VRAM.

---

## 4. The sequence model: BiLSTM

From [model.py](src/model.py):

```python
self.rnn = nn.LSTM(
    input_size=512, hidden_size=256, num_layers=2,
    bidirectional=True, batch_first=True,
)
```

After the CNN, we have one 512-dimensional feature vector per horizontal
position — but those vectors were computed independently of each other
(a convolution only sees a local neighborhood, its *receptive field*).
Reading Arabic depends on context: the same rasterized shape can belong
to different letters depending on what's connected before/after it. The
BiLSTM lets every position's final representation depend on the *entire*
line, read in both directions.

**How LSTM handles the "context" problem**: a plain RNN processes a
sequence step by step, carrying a hidden state forward, but has a hard
time keeping information from many steps ago (the vanishing gradient
problem — gradients shrink exponentially over long distances during
backpropagation through time). LSTM (Long Short-Term Memory) fixes this
with gates — an input gate, forget gate, and output gate control what
information gets written into, kept in, or read out of a memory cell,
allowing gradients to flow much further back without vanishing.
**Bidirectional** just means we run one LSTM left-to-right and a
separate one right-to-left, then concatenate their outputs at each
position — so every position sees both what came before *and* after it.

### Alternatives considered

| Option | Idea | Why not used |
|---|---|---|
| **GRU** (Gated Recurrent Unit) | Simplified LSTM — merges the forget/input gates into one "update gate," fewer parameters | Roughly comparable accuracy to LSTM on most tasks, trains slightly faster; genuinely a reasonable alternative. LSTM was picked mainly because it's the standard in the reference CRNN architecture and the difference is marginal at this scale — not a strong technical reason either way. |
| **Self-attention / Transformer encoder** | Every position attends to every other position directly, no recurrence | Would work, and is what modern OCR models increasingly use, but needs more data/compute to train from scratch well, and adds complexity (positional encodings, more hyperparameters) with unclear benefit at this dataset size |
| **Pure CNN (fully convolutional, no recurrence)** | Stack more conv layers with a big enough receptive field instead of an RNN | Genuinely used in some fast OCR systems (e.g. some PaddleOCR configs); trades some accuracy for speed. We had no strict latency requirement, so we kept the RNN for its stronger context modeling. |

---

## 5. CTC loss — the deepest part of this project

This is worth explaining properly, since it's the piece that makes the
whole "no per-character alignment needed" idea actually work.

### The core idea

At every one of the `T` time steps output by the CNN+BiLSTM, the model
outputs a probability distribution over the vocabulary (including a
special `blank` symbol — see [vocab.py](src/vocab.py), `BLANK_IDX = 0`).
A raw output might look like:

```
h h h - e e - - l l - l - o o -      (- = blank)
```

CTC defines a deterministic **collapse rule** to turn this into final
text: merge consecutive repeats, then delete all blanks. The example
above collapses to `"hello"`. Critically, the blank symbol is what makes
it possible to have `"hello"` (with a repeated `l`) instead of the
repeat-collapsing rule accidentally merging the two `l`s into one — you
put a blank between them: `l l - l -` → collapses to `l l`, i.e. two
l's survive because the blank breaks the repeat-merge.

### Why this solves the alignment problem

For a given label like `"hello"`, there are **many** different raw
`T`-length sequences that all collapse to `"hello"` — the letters can be
"stretched" over different numbers of time steps, and blanks can be
inserted in different places. CTC loss doesn't pick one specific
alignment and penalize deviation from it. Instead, it sums the
probabilities of *every possible* raw sequence that collapses to the
correct label, and the loss is the negative log of that total
probability. This sum, over an exponential number of possible
alignments, is computed efficiently using **dynamic programming** — the
forward-backward algorithm (closely related to the algorithm used in
Hidden Markov Models). It builds a lattice/trellis over (time step,
position in an "expanded" label with blanks inserted between every
character) and accumulates path probabilities, so the whole thing runs
in time proportional to `T × label_length` rather than exponential time.

In our code, this is one line, `nn.CTCLoss` in
[train.py](src/train.py):

```python
criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
...
loss = criterion(log_probs, targets, input_lengths, target_lengths)
```

`zero_infinity=True` matters practically: if a target is longer than
what's mathematically representable in the available time steps (not
enough room even with blanks), the loss would become infinite and
poison the gradient for the whole batch; this flag zeroes out that one
sample's contribution instead of crashing training.

### Why `input_lengths` exists — and how our width formula came from it

Images in a batch are padded to the same width (see [dataset.py](src/dataset.py),
`collate_fn`), but padded columns aren't real content. CTC needs to know
how many of the `T` output steps are "real" per sample. Since our CNN
downsamples width deterministically (three stride-2 pooling operations,
then a valid 2×2 conv that trims one column), we derive:

```python
@staticmethod
def compute_output_width(width: int) -> int:
    return max(1, width // 4 - 1)
```

directly from the padding/stride math of the CNN — this is what lets
`train.py` tell `CTCLoss` (and `infer.py`/`infer_page.py` tell the
greedy decoder) exactly which output columns are meaningful.

### Alternatives to CTC

| Approach | Idea | Why not used |
|---|---|---|
| **Cross-entropy with forced alignment** | Get a per-character alignment first (e.g. via a separate alignment model, or manual labeling), then train a normal per-position classifier | Requires alignment data we don't have and don't want to manually produce — defeats the purpose of using synthetic weak labels |
| **Attention seq2seq cross-entropy** | Decoder attends and predicts one character at a time, trained with per-step cross-entropy + teacher forcing | See section 2 — more complex, slower inference, more data-hungry |
| **RNN-Transducer loss** | Like CTC but the predictor also conditions on previous outputs (not just the input) | See section 2 — extra complexity not justified by our accuracy needs |

CTC was the right choice because our problem — turn an unsegmented image
into unsegmented text, with a roughly monotonic left-to-right
correspondence between the two — is exactly the problem CTC was
invented for (originally for speech recognition, at Alex Graves et al.,
2006, later widely adopted for OCR/handwriting recognition).

---

## 6. Decoding: turning probabilities into text

[decode.py](src/decode.py) implements **greedy decoding**: take the
`argmax` character at every time step, then apply the same
collapse-repeats-then-remove-blanks rule described above.

### Alternatives

| Method | Idea | Trade-off |
|---|---|---|
| **Greedy (used)** | Take the single most likely character at every step independently | Fast, simple, but can miss the *globally* best sequence — a locally suboptimal choice at one step can't be "corrected" later |
| **Beam search** | Keep the top-`k` most likely partial sequences at every step instead of just 1, expand each, keep the best `k` again | More accurate, especially when the model is genuinely unsure between two close options; standard practice for production OCR. Costs more compute at inference. |
| **Beam search + language model** | Same as above, but rescore beams using an external n-gram or neural language model over Arabic text, so linguistically implausible outputs are penalized | Would meaningfully help with our diacritics/substitution errors (the model sometimes "sounds visually right but linguistically wrong"). Not implemented here — a clear next step. |

We used greedy decoding because it's what's needed to validate the
pipeline quickly; upgrading to beam search (or beam search + a small
Arabic language model) is a well-understood, incremental improvement
that doesn't require retraining the recognizer itself.

---

## 7. Data: synthetic generation vs. alternatives

[generate_data.py](src/generate_data.py) renders random sentences
([corpus.py](src/corpus.py)) with random fonts/sizes and light
augmentation (rotation, Gaussian blur, Gaussian noise, contrast jitter).

### Alternatives

| Source | Pros | Cons | Why synthetic was primary |
|---|---|---|---|
| **Synthetic rendering (used)** | Unlimited volume, perfect labels (we generated the text, so we know it exactly), fast to produce (200k images in ~7 minutes) | Doesn't capture real scan noise, real fonts, real layout — this is exactly the "domain gap" seen in section 12 | Only realistic option to bootstrap a from-scratch model with zero existing labeled Arabic OCR data on hand |
| **Public datasets** (KHATT — handwriting, APTI — printed Arabic, ADAB, PATS-A01...) | Real handwriting/print statistics, established benchmarks | Need to be sourced/downloaded, often licensing-restricted, not infinite, may not match your specific target documents (e.g. this specific book's font) | Good complement, not a full replacement — mentioned as a next step in the README |
| **Crowdsourced labeling** | Real labels for real target documents | Slow, costly, needs infrastructure | Not practical for a solo/from-scratch project |
| **Semi-supervised / self-training** | Run the current model on unlabeled real data, keep high-confidence predictions as pseudo-labels, retrain | Works well once you have a *decent* base model | Reasonable follow-up once the current ~96% synthetic / ~75-80% real accuracy baseline exists — you could pseudo-label more real pages and iterate |

### Augmentation alternatives not used

- **Elastic distortion** (warps the image with a smooth random
  displacement field) — common for handwriting-style augmentation,
  skipped here since our target is mainly printed text.
- **Cutout / random erasing** (blank out random rectangular patches) —
  useful for robustness to occlusion, less relevant for clean scans.
- **GAN-based augmentation** (a generative model learns to produce
  realistic-looking degraded text) — powerful but a project of its own;
  far beyond what a first version needs.

---

## 8. Arabic text shaping

To render Arabic correctly, [generate_data.py](src/generate_data.py) does:

```python
reshaped = arabic_reshaper.reshape(text)   # select correct glyph per position
visual_text = get_display(reshaped)         # reorder into left-to-right visual order
```

**Why this is needed at all**: Unicode stores Arabic text in *logical*
order (the order you'd type it, right to left conceptually) using
*base* letter codepoints. But a rendered letter's *shape* depends on
whether it's isolated, or joins to a letter before/after it (up to 4
glyph variants per letter). Turning logical text into the correct visual
glyphs is called **text shaping**.

### Alternatives

| Approach | How it works | Why not the primary path here |
|---|---|---|
| **arabic-reshaper + python-bidi (used)** | Pure-Python: reshaper swaps each character for the correct presentation-form codepoint based on its neighbors; python-bidi reorders the string for LTR rendering, implementing the Unicode Bidirectional Algorithm | Lightweight, no native dependencies, works with any Pillow install |
| **Raqm / HarfBuzz** (a proper text-shaping engine) | Native library that does shaping *and* complex-script layout (ligatures, kerning, mark positioning) at the font level, used by real browsers/OS text renderers | Higher quality shaping (handles more edge cases correctly), but requires Pillow to be compiled with `libraqm` support. We checked: `PIL.features.check('raqm')` returned `False` on this machine, so we used the fallback. This is noted directly in the code path. |

The practical effect: `arabic-reshaper` + `python-bidi` is "good enough"
and gave clean, correctly-joined renders (verified visually during
generation), but a HarfBuzz-backed renderer would in principle produce
slightly more typographically accurate output (better handling of rare
ligatures, kerning). Not worth chasing for this project's accuracy
bottleneck, which lies elsewhere (data domain gap, not glyph shaping).

---

## 9. Vocabulary design

[vocab.py](src/vocab.py) makes two specific design choices worth calling
out:

**1. Character-level, not presentation-form-level.** We could have made
the CTC vocabulary include each letter's 4 shape variants as separate
classes (since that's literally what's drawn on screen). We didn't —
the vocabulary only has *base* letters. The CNN+BiLSTM is trusted to
learn "these 4 visually different glyphs all mean the same underlying
letter" implicitly, the same way a human reader does. This keeps the
vocabulary small (~90 symbols instead of 300+), which means a smaller,
easier-to-train final classification layer.

**2. Logical order, with a reversal trick for CTC targets.**

```python
def text_to_ctc_indices(text: str) -> list[int]:
    reversed_text = text[::-1]
    return [CHAR_TO_IDX[c] for c in reversed_text if c in CHAR_TO_IDX]
```

Arabic reads right-to-left, but the CNN scans image columns left-to-right
(there's no "scan right-to-left" option in a standard conv/pooling
stack without extra plumbing). Rather than changing how the image is
scanned, we simply reverse the *label* string so its order matches the
image's left-to-right visual order. This is a simplification that
assumes lines are "purely" RTL (no embedded Latin/LTR runs beyond
isolated digit groups) — see the caveat in the README about richer
bidi mixing needing real bidi-aware alignment instead of a blind
reversal.

---

## 10. Training infrastructure

### Framework: PyTorch

Chosen over **TensorFlow/Keras** or **JAX** mainly because: dynamic-graph
debugging is easier (you can inspect intermediate tensors like normal
Python objects), `nn.CTCLoss` and variable-length RNN handling are
mature and well-documented, and it's the dominant framework for this
kind of research-style project today. TensorFlow would work fine too —
this was a workflow preference, not a hard technical requirement. JAX is
excellent for large-scale/research-heavy setups (functional
transformations, easy multi-device scaling) but adds unnecessary
functional-programming overhead for a project this size.

### Optimizer: Adam

```python
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
```

Adam keeps a running per-parameter estimate of both the gradient's mean
and variance, and uses them to adapt the effective learning rate per
parameter — this makes it forgiving of imperfect learning-rate tuning
and converges quickly, which matters when you have limited time to
iterate on a laptop.

**Alternatives**: **SGD with momentum** often reaches slightly better
final accuracy for image classification given enough tuning and training
time, but needs careful learning-rate scheduling to get there — too slow
to tune for a from-scratch first pass. **AdamW** (Adam with decoupled
weight decay) is arguably a strictly better default than Adam in modern
practice; a reasonable upgrade to try, but we didn't use any weight
decay here in the first place, so it wouldn't have changed anything yet.

### Mixed precision (AMP)

```python
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
...
with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
    log_probs = model(images)
    loss = criterion(...)
scaler.scale(loss).backward()
```

Runs most operations in 16-bit floating point instead of 32-bit —
roughly half the memory, and faster on GPUs with tensor cores (the RTX
3050 has them). The `GradScaler` exists because 16-bit floats have a
much smaller representable range, so gradients can underflow to zero;
the scaler multiplies the loss by a large factor before the backward
pass and unscales the gradients after, avoiding that underflow.

**Alternatives**: **Full FP32** is simplest and safest but uses ~2x the
memory and is slower — would have forced a smaller batch size on a 4GB
card. **Pure FP16** (no scaler) risks silent underflow/instability.
**BF16** (bfloat16) has FP32's exponent range so it doesn't need a loss
scaler at all, but isn't as reliably supported/accelerated on all
consumer GPU generations as FP16 — AMP with FP16+GradScaler is the
safer default choice for this hardware.

### Batching: pad-to-max-width vs. bucketing

```python
padded = torch.ones(len(images), 1, height, max_w, dtype=torch.float32)
```

Every image in a batch is right-padded to that batch's longest image.

**Alternative — length bucketing**: sort/group samples by similar
lengths before batching, so a batch of naturally-short lines isn't
forced to pad up to one long outlier. This reduces wasted compute on
padding columns. We didn't implement it — with random shuffling and a
batch size of 64, the padding overhead was acceptable for this dataset's
length distribution, but it's a legitimate speed optimization for a
larger run.

---

## 11. Page-level OCR: line segmentation

[segment_lines.py](src/segment_lines.py) uses a **horizontal ink
projection profile** — classical computer vision, not machine learning:

```python
row_ink = ink_mask.sum(axis=1) / w      # fraction of ink per row
is_text_row = row_ink > min_ink_ratio   # is this row "inside" a line?
```

Rows with enough dark pixels are "inside" a text line; a long enough run
of near-empty rows marks a gap between lines. After finding a line's
vertical band, we additionally crop it horizontally to the actual ink's
bounding box (not the full page width) — this fix mattered in practice:
without it, short lines got padded with huge blank margins the
recognizer had never seen during training, which caused it to
hallucinate repeated garbage characters into that blank space (see the
before/after test earlier in this project).

### Alternatives

| Method | Idea | Why not used (yet) |
|---|---|---|
| **Projection profile (used)** | Pure pixel-counting, no training needed | Fast, zero-cost, works well on clean, non-skewed, single-column scans — exactly what our test pages are. Breaks on skewed/curved/multi-column/noisy scans. |
| **Deep text detectors** (CRAFT, DBNet, EAST) | Neural networks trained to output per-character or per-word/line bounding regions directly from pixels, robust to skew/curve/clutter | Would handle messier real-world documents far better, but is its own separate model that needs its own training data and time — deferred as future work, not needed for clean scanned book pages |
| **Document layout analysis models** (e.g. LayoutParser, Mask R-CNN-based) | Understand full page structure: paragraphs, columns, footnotes, headers, reading order | The "proper" solution for complex multi-column academic/legal documents; heavier infrastructure than this project currently needs |
| **Tesseract's built-in page segmentation** | Tesseract ships its own layout analysis (page segmentation modes) | Would mean depending on Tesseract for the segmentation step while still using our own recognizer — a valid hybrid, not pursued since the classical method already worked on our target pages |

---

## 12. The synthetic-to-real gap

Measured directly: **~4% character error rate (CER)** on synthetic
validation data, **~20-25% CER** on real scanned book pages. Why the gap
exists, precisely:

1. **Font mismatch** — training used 6 system fonts (Arial, Tahoma,
   Segoe UI, Arabic Typesetting); the real book uses a different print
   typeface entirely, with different letterforms, stroke widths, and
   diacritic placement.
2. **Layout mismatch** — training lines are short, isolated, unjustified
   sentences; real book lines are long, justified (stretched to fill a
   column width, changing inter-word/inter-letter spacing), sometimes
   bold/mixed-weight.
3. **Diacritics are visually tiny** — a few pixels tall at normal scan
   resolution, easy for both synthetic rendering and the model to get
   slightly wrong; this alone accounts for a large share of the error.
4. **Real scan artifacts** — actual print/scan noise differs
   systematically from our synthetic Gaussian-noise approximation.

### How to close it (not yet implemented, but the correct next steps)

- **Fine-tuning on real data**: manually transcribe a modest number of
  real lines (even a few hundred), and continue training
  (`--resume`) on a mix of synthetic + real data. Usually the single
  biggest lever.
- **Style-matched synthetic data**: identify the book's actual font (or
  a close visual match) and regenerate synthetic data using it, with
  justified-paragraph-style layout instead of short isolated sentences.
- **Domain adaptation techniques** (more advanced, not needed yet): e.g.
  a Domain-Adversarial Neural Network (DANN) trains the feature
  extractor to produce features that a separate "domain classifier"
  *can't* tell apart between synthetic and real images — pushing the
  CNN to learn domain-invariant features. Overkill until the simpler
  fine-tuning route is exhausted.

---

## 13. File-by-file code map

| File | Role |
|---|---|
| [vocab.py](src/vocab.py) | Character ↔ index mapping; logical/CTC-order conversion |
| [corpus.py](src/corpus.py) | Word list + random sentence/word/number generators |
| [generate_data.py](src/generate_data.py) | Renders text → line images, applies augmentation, writes `labels.tsv` |
| [dataset.py](src/dataset.py) | PyTorch `Dataset` + `collate_fn` (variable-width padding) |
| [model.py](src/model.py) | CRNN definition (CNN → BiLSTM → Linear) |
| [decode.py](src/decode.py) | Greedy CTC decoding |
| [train.py](src/train.py) | Training loop, CTC loss, AMP, checkpointing, CER evaluation |
| [infer.py](src/infer.py) | Single-line inference from a saved checkpoint |
| [segment_lines.py](src/segment_lines.py) | Classical line segmentation for full pages |
| [infer_page.py](src/infer_page.py) | segmentation + per-line recognition, chained |

---

## 14. Honest limitations

- No beam search / language model decoding yet — greedy decoding leaves
  accuracy on the table, especially for diacritics.
- No real training data yet — everything the model has ever seen is
  synthetic.
- Line segmentation is classical CV, not learned — will struggle on
  skewed, curved, multi-column, or heavily degraded scans.
- No handling of tables, images-in-text, or non-standard reading order.
- The bidi/label-reversal scheme assumes mostly-pure-RTL lines; heavy
  Latin/RTL mixing within a line isn't handled rigorously.
