# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration of the low-bit draft vocab embedding (A1c) into the spec config
and the draft load path.

The numerical core (QuantizedVocabEmbedding) is covered by
test_quantized_draft_embedding.py. Here we test (1) the SpeculativeConfig knob
that selects the draft embedding bit-width, and (2) the proposer swapping the
draft's own embedding for a quantized one on the PP separate-load path.

CPU-only: config plumbing + a stubbed inner model, no download / distributed init.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.quantized_draft_embedding import (
    QuantizedVocabEmbedding,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

PP = "vllm.v1.spec_decode.llm_base_proposer.get_pp_group"


def _stub_proposer(bits, embed):
    """A minimal stand-in exposing only what _maybe_quantize_draft_embed reads."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(draft_embed_quant_bits=bits)
        ),
        model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed)),
    )


# --- Cycle 1: the config knob -------------------------------------------------


def test_verify_draft_embed_quant_bits_defaults_none():
    """Unset -> None (no quantization, full-precision draft embed)."""
    assert SpeculativeConfig._verify_draft_embed_quant_bits(None) is None


def test_verify_draft_embed_quant_bits_allows_8_and_4():
    assert SpeculativeConfig._verify_draft_embed_quant_bits(8) == 8
    assert SpeculativeConfig._verify_draft_embed_quant_bits(4) == 4


def test_verify_draft_embed_quant_bits_rejects_other_values():
    """Only 4 or 8 (or None) are supported bit-widths."""
    with pytest.raises(ValueError):
        SpeculativeConfig._verify_draft_embed_quant_bits(3)
    with pytest.raises(ValueError):
        SpeculativeConfig._verify_draft_embed_quant_bits(16)


# --- Cycle 2: the proposer swaps in a quantized draft embed under PP -----------


def test_swaps_in_quantized_embed_under_pp():
    """With bits set and PP>1 (draft has its own embed), the draft's embedding
    becomes a QuantizedVocabEmbedding whose lookup tracks the original."""
    torch.manual_seed(0)
    vocab, hidden = 200, 32
    embed = torch.nn.Embedding(vocab, hidden, dtype=torch.float16)
    proposer = _stub_proposer(bits=8, embed=embed)

    with patch(PP, return_value=SimpleNamespace(world_size=2)):
        SpecDecodeBaseProposer._maybe_quantize_draft_embed(proposer)

    new_embed = proposer.model.model.embed_tokens
    assert isinstance(new_embed, QuantizedVocabEmbedding)

    ids = torch.tensor([0, 7, 199], dtype=torch.long)
    ref = torch.nn.functional.embedding(ids, embed.weight)
    out = new_embed(ids)
    assert (out.float() - ref.float()).abs().max().item() < 0.03


def test_no_quantization_when_bits_none():
    """bits=None leaves the draft embedding untouched."""
    embed = torch.nn.Embedding(50, 16, dtype=torch.float16)
    proposer = _stub_proposer(bits=None, embed=embed)

    with patch(PP, return_value=SimpleNamespace(world_size=2)):
        SpecDecodeBaseProposer._maybe_quantize_draft_embed(proposer)

    assert proposer.model.model.embed_tokens is embed


def test_no_quantization_without_pp():
    """Under world_size==1 the embed is shared with the target; never mutate it."""
    embed = torch.nn.Embedding(50, 16, dtype=torch.float16)
    proposer = _stub_proposer(bits=8, embed=embed)

    with patch(PP, return_value=SimpleNamespace(world_size=1)):
        SpecDecodeBaseProposer._maybe_quantize_draft_embed(proposer)

    assert proposer.model.model.embed_tokens is embed
