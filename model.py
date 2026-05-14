"""
BiLSTM + Additive Attention sleep staging model — v3.

Changes from v2 (centre-extraction only):
  ● Additive self-attention over all LSTM positions
  ● Centre hidden state concatenated with attention context → richer representation
  ● This combination helps N1 detection:
      - Centre h  captures the epoch's own spectral profile
      - Attention h can focus on the TRANSITION CONTRAST around short N1 bursts
        (61 % of N1 runs are ≤2 epochs — the contrast with surrounding Wake/N2
         is more discriminative than any single-epoch feature)

Architecture:
    Input  (B, 45, 10)
      │
    LayerNorm(10)
      │
    BiLSTM ×2 layers         (B, 45, 512)   [256 per direction]
      │                              │
      │   h_centre = h[:, 22, :]    │  attn weights = softmax(Wh)
      │        (B, 512)             │  context = Σ attn * h
      │                             │        (B, 512)
      └──────────── cat ────────────┘
                    (B, 1024)
      │
    LayerNorm(1024) + Dropout
      │
    Linear(1024 → 5)          logits

Total params: ~2.5M  (up from 2.1M; 16 GB VRAM: trivial overhead)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CENTER_IDX, DROPOUT, HIDDEN_SIZE, INPUT_SIZE, NUM_CLASSES, NUM_LAYERS


class BiLSTMSleepStager(nn.Module):

    def __init__(
        self,
        input_size:  int   = INPUT_SIZE,
        hidden_size: int   = HIDDEN_SIZE,
        num_layers:  int   = NUM_LAYERS,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = DROPOUT,
    ) -> None:
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Additive attention: single linear layer maps each hidden state
        # to a scalar score, then softmax normalises over the sequence.
        # bias=False: scores are purely based on the hidden state direction.
        self.attn_q = nn.Linear(hidden_size * 2, 1, bias=False)

        # Input to classifier = centre h  ‖  attention context
        # hidden_size * 4 = 2 (bidirectional) × 2 (centre + context)
        self.out_norm   = nn.LayerNorm(hidden_size * 4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 4, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)   # forget gate = 1
        nn.init.xavier_uniform_(self.attn_q.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, WINDOW_SIZE, input_size)
        returns: (B, num_classes) — raw logits
        """
        x = self.input_norm(x)                            # (B, W, D)
        lstm_out, _ = self.lstm(x)                        # (B, W, H*2)

        # Centre hidden state: full bilateral context at the target epoch
        h_centre = lstm_out[:, CENTER_IDX, :]             # (B, H*2)

        # Attention context: let the model choose which positions to emphasise.
        # For short N1 bursts (1–2 epochs), the contrast at surrounding positions
        # (Wake before, N2 after) carries strong discriminative signal.
        attn_logits  = self.attn_q(lstm_out).squeeze(-1)  # (B, W)
        attn_weights = F.softmax(attn_logits, dim=1)      # (B, W)
        h_context    = (attn_weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (B, H*2)

        # Concatenate: preserve both the centre representation and the
        # flexible attended context — the classifier learns to weight them.
        combined = torch.cat([h_centre, h_context], dim=1)  # (B, H*4)
        combined = self.out_norm(combined)

        return self.classifier(combined)                  # (B, C)
