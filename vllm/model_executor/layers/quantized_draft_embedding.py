# SPDX-License-Identifier: Apache-2.0
"""Low-bit vocab embedding for speculative-decode draft models (A1c).

The MTP draft loads its own full-precision copy of a huge vocab embedding on the
last PP rank, which is what pushes 27B+draft over 2x16 GiB. Because the draft is
only a speculative proposer (rejection sampling guarantees the final output equals
target greedy regardless of draft quality), a *lossy* low-bit draft embedding is
correctness-safe: it only lowers acceptance, never changes the output.

This module stores the embedding at a lower bit-width and dequantizes the
looked-up rows on the fly. Targets TP=1 (the draft runs single-rank), so the
lookup is a plain gather with no all-reduce.

Bit-widths:
- ``bits=8``: per-row symmetric int8, 1 byte/weight (~2x saving).
- ``bits=4``: per-row symmetric int4 packed 2 nibbles/byte, 0.5 byte/weight
  (~4x saving). Requires an even hidden dimension.
"""

import torch
from torch import nn


class QuantizedVocabEmbedding(nn.Module):
    """Per-row symmetric low-bit vocab embedding (lookup + dequant).

    Args:
        weight: the full-precision ``[vocab, hidden]`` embedding to quantize.
        bits: quantization bit-width (8 or 4).
    """

    def __init__(self, weight: torch.Tensor, *, bits: int = 8) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise NotImplementedError(f"bits={bits} not yet supported")
        self.bits = bits
        self.out_dtype = weight.dtype
        self.hidden_size = weight.shape[1]

        qmax = 2 ** (bits - 1) - 1  # 127 (int8) / 7 (int4)
        # Per-row (per-token) symmetric scale.
        row_absmax = weight.abs().amax(dim=1).clamp_min(1e-12)
        scale = (row_absmax / qmax).to(torch.float32)
        q = (weight.to(torch.float32) / scale.unsqueeze(1)).round().clamp_(-qmax, qmax)

        if bits == 8:
            qweight = q.to(torch.int8)
        else:  # bits == 4: pack two signed nibbles per uint8 byte
            if self.hidden_size % 2 != 0:
                raise ValueError("int4 packing requires an even hidden size")
            u = (q + 8).to(torch.uint8)  # [-7,7] -> [1,15], fits 4 bits
            low = u[:, 0::2]
            high = u[:, 1::2]
            qweight = (low | (high << 4)).contiguous()

        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = self.qweight[input_ids]
        if self.bits == 8:
            deq = rows.to(torch.float32)
        else:  # unpack nibbles back to signed int4 values
            low = (rows & 0x0F).to(torch.int16) - 8
            high = ((rows >> 4) & 0x0F).to(torch.int16) - 8
            deq = torch.stack([low, high], dim=-1)
            deq = deq.reshape(*rows.shape[:-1], self.hidden_size).to(torch.float32)
        out = deq * self.scale[input_ids].unsqueeze(-1)
        return out.to(self.out_dtype)
