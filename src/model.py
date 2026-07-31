"""CRNN (CNN + BiLSTM) model for CTC-based Arabic text-line recognition.

Expects input images of fixed height 32 (see dataset.py). The CNN reduces
the height dimension to 1 and downsamples width by a factor of 4 (minus a
constant), producing a sequence of feature vectors fed to a BiLSTM stack
and a final linear classifier over the character vocabulary.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, vocab_size: int, lstm_hidden: int = 256, lstm_layers: int = 2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, W/4

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4

            nn.Conv2d(512, 512, 2, padding=0),  # H/16-1 -> 1, W/4-1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H=32, W) -> log_probs: (T, B, vocab_size)"""
        features = self.cnn(x)  # (B, C, H', W')
        b, c, h, w = features.shape
        assert h == 1, f"expected CNN output height 1, got {h} (input height must be 32)"
        features = features.squeeze(2)  # (B, C, W')
        features = features.permute(0, 2, 1)  # (B, W', C)

        seq, _ = self.rnn(features)  # (B, W', 2*hidden)
        logits = self.fc(seq)  # (B, W', vocab_size)
        log_probs = logits.log_softmax(dim=2)

        return log_probs.permute(1, 0, 2)  # (T=W', B, vocab_size)

    @staticmethod
    def compute_output_width(width: int) -> int:
        """Maps input image width -> output sequence length (T), matching
        the CNN's deterministic downsampling (stride-4 pooling, then a
        valid 2x2 conv that trims 1)."""
        return max(1, width // 4 - 1)
