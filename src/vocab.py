"""Character vocabulary for the Arabic OCR model.

Labels are stored in *logical* reading order (normal Arabic string order).
CTC targets are derived from this in reversed order to match the left-to-right
scan direction of the image (see dataset.py / generate_data.py).
"""

ARABIC_LETTERS = list(
    "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
)

DIACRITICS = list("ًٌٍَُِّْٰ")

DIGITS_ARABIC_INDIC = list("٠١٢٣٤٥٦٧٨٩")
DIGITS_WESTERN = list("0123456789")

PUNCTUATION = list("،؛؟!.:-()\"'")

SPACE = " "

# CTC blank must be index 0.
BLANK = "<blank>"

CHARS = (
    [BLANK, SPACE]
    + ARABIC_LETTERS
    + DIACRITICS
    + DIGITS_ARABIC_INDIC
    + DIGITS_WESTERN
    + PUNCTUATION
)

CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}

VOCAB_SIZE = len(CHARS)
BLANK_IDX = CHAR_TO_IDX[BLANK]


def text_to_ctc_indices(text: str) -> list[int]:
    """Logical-order text -> CTC target indices, reversed to match the
    left-to-right visual scan order of the rendered image."""
    reversed_text = text[::-1]
    return [CHAR_TO_IDX[c] for c in reversed_text if c in CHAR_TO_IDX]


def ctc_indices_to_text(indices: list[int]) -> str:
    """Inverse of text_to_ctc_indices: predicted left-to-right indices ->
    logical-order Arabic text."""
    chars = [IDX_TO_CHAR[i] for i in indices]
    return "".join(chars)[::-1]


if __name__ == "__main__":
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Chars: {CHARS}")
