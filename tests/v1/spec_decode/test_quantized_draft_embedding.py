# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the low-bit draft vocab embedding (A1c memory fix).

The MTP draft loads its own full-precision copy of a huge vocab embedding on the
last PP rank (Qwen3.5: vocab 248320 x hidden 5120 = 2.37 GiB fp16), which is what
pushes 27B+draft over 2x16 GiB. Because the draft is only a speculative proposer
(rejection sampling guarantees the final output equals target greedy regardless of
draft quality), a *lossy* low-bit draft embedding is correctness-safe: it only
lowers acceptance, never changes the output. These tests pin the numerical core:
a low-bit embedding lookup must approximate the fp16 lookup within quant error,
while storing weights at the target bit-width.

Pure-torch (CPU), no GPU / PP / model needed.
"""

import torch

from vllm.model_executor.layers.quantized_draft_embedding import (
    QuantizedVocabEmbedding,
)


def test_int8_lookup_matches_fp16_within_quant_error():
    """int8 per-row symmetric: lookup ~= fp16 lookup within one quant step."""
    torch.manual_seed(0)
    vocab, hidden = 1000, 64
    weight = torch.randn(vocab, hidden, dtype=torch.float16)

    qemb = QuantizedVocabEmbedding(weight, bits=8)

    ids = torch.tensor([0, 5, 42, 999, 500], dtype=torch.long)
    ref = torch.nn.functional.embedding(ids, weight)
    out = qemb(ids)

    assert out.shape == ref.shape
    assert out.dtype == torch.float16
    # Per-row symmetric int8: scale = row_absmax / 127, so the worst-case
    # per-element error is scale/2 = row_absmax/254. For ~N(0,1) rows this is
    # well under 0.03; assert that bound.
    max_err = (out.float() - ref.float()).abs().max().item()
    assert max_err < 0.03, f"int8 lookup error too large: {max_err}"


def test_int8_stores_one_byte_per_weight():
    """The win: int8 weight storage is half of fp16, plus a tiny per-row scale."""
    torch.manual_seed(0)
    vocab, hidden = 1000, 64
    weight = torch.randn(vocab, hidden, dtype=torch.float16)

    qemb = QuantizedVocabEmbedding(weight, bits=8)

    assert qemb.qweight.dtype == torch.int8
    assert qemb.qweight.numel() == vocab * hidden
    # One scale per row (per-row quantization).
    assert qemb.scale.numel() == vocab


def test_int4_lookup_matches_fp16_within_quant_error():
    """int4 is coarser than int8 but still tracks the fp16 lookup."""
    torch.manual_seed(0)
    vocab, hidden = 1000, 64
    weight = torch.randn(vocab, hidden, dtype=torch.float16)

    qemb = QuantizedVocabEmbedding(weight, bits=4)

    ids = torch.tensor([0, 5, 42, 999, 500], dtype=torch.long)
    ref = torch.nn.functional.embedding(ids, weight)
    out = qemb(ids)

    assert out.shape == ref.shape
    assert out.dtype == torch.float16
    # int4 per-row symmetric: scale = row_absmax / 7, worst-case per-element
    # error scale/2 = row_absmax/14 (~0.28 for ~N(0,1) rows). Bound at 0.5.
    max_err = (out.float() - ref.float()).abs().max().item()
    assert max_err < 0.5, f"int4 lookup error too large: {max_err}"
    # Sanity: genuinely coarser than int8 would be.
    assert max_err > 0.03, f"int4 unexpectedly precise: {max_err}"


def test_int4_packs_two_weights_per_byte():
    """The bigger win: int4 stores half a byte per weight (2 nibbles/byte)."""
    torch.manual_seed(0)
    vocab, hidden = 1000, 64
    weight = torch.randn(vocab, hidden, dtype=torch.float16)

    qemb = QuantizedVocabEmbedding(weight, bits=4)

    assert qemb.qweight.dtype == torch.uint8
    assert qemb.qweight.numel() == vocab * hidden // 2
    assert qemb.scale.numel() == vocab
