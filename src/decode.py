"""Greedy CTC decoding shared by training (for CER eval) and inference."""

import torch

from vocab import BLANK_IDX, ctc_indices_to_text


def greedy_decode_indices(log_probs: torch.Tensor, input_lengths: torch.Tensor) -> list[list[int]]:
    """log_probs: (T, B, V). Returns per-sample collapsed CTC index lists
    (blanks removed, repeats collapsed), still in left-to-right visual
    order (i.e. not yet reversed to logical order)."""
    preds = log_probs.argmax(dim=2)  # (T, B)
    preds = preds.transpose(0, 1)  # (B, T)

    results = []
    for b in range(preds.shape[0]):
        length = int(input_lengths[b])
        seq = preds[b, :length].tolist()

        collapsed = []
        prev = None
        for idx in seq:
            if idx != prev and idx != BLANK_IDX:
                collapsed.append(idx)
            prev = idx
        results.append(collapsed)
    return results


def greedy_decode_texts(log_probs: torch.Tensor, input_lengths: torch.Tensor) -> list[str]:
    """Same as greedy_decode_indices but returns logical-order Arabic strings."""
    index_lists = greedy_decode_indices(log_probs, input_lengths)
    return [ctc_indices_to_text(idxs) for idxs in index_lists]
