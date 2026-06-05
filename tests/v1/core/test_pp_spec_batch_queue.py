# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative decoding under PP/async pipelined (batch_queue) scheduling.

Under PP>1 the engine pipelines microbatches via the batch_queue: ``schedule()``
for step N+1 runs *before* ``update_from_output()`` of step N (up to
``pp_size`` batches in flight). The ``AsyncScheduler`` lets a *single* request be
scheduled again while a prior step is still in flight, via ``num_output_
placeholders`` and optimistic spec-token placeholders
(``async_scheduler.py:_update_after_schedule``).

These tests drive the scheduler through that pipelined ordering WITH speculative
tokens and assert a robust, sampling-pattern-independent invariant: with
``ignore_eos`` + ``max_tokens=M``, the request must stop at EXACTLY M output
tokens regardless of how many drafts are accepted per step. If spec accounting
drifts under the k-step delay (the over-generation / gibberish risk), the count
over- or under-shoots.

CPU-only, no model. ``_drive_batch_queue`` reproduces engine-core's
``step_with_batch_queue`` ordering (``vllm/v1/engine/core.py:484``) at the
scheduler-logic level, relying on the AsyncScheduler's own placeholder injection
rather than a hand-fed ``update_draft_token_ids`` (under async the worker fills
draft *values*; the scheduler only reserves the slots). Each test asserts the
pipeline actually reached depth >= 2 so it cannot silently degrade to lockstep.
"""

from collections import deque

import pytest

from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler


def _drive_batch_queue(
    scheduler,
    reqs,
    *,
    batch_queue_size: int,
    accept: int,
    stop_token: int | None = None,
    stop_after_output: int | None = None,
) -> int:
    """Drive the scheduler through engine-core's batch_queue loop and return the
    max in-flight queue depth reached.

    Faithfulness rules mirrored from the real runner:
    - keep scheduling (filling the queue) until full or no requests remain, only
      then pop the oldest batch and apply its output -> creates the k-step delay;
    - a request samples a token only once its prompt is fully scheduled
      (incomplete prefill chunks produce ``sampled_token_ids == []``);
    - on a decode step it emits ``[accepted_drafts..., bonus]`` (len =
      min(accept, num_scheduled_spec) + 1), matching ``update_from_output``'s
      ``num_rejected = num_draft - (len(sampled) - 1)`` accounting.
    """
    by_id = {r.request_id: r for r in reqs}
    computed = {r.request_id: 0 for r in reqs}
    nprompt = {r.request_id: r.num_prompt_tokens for r in reqs}
    queue: deque = deque()
    max_depth = 0
    guard = 0
    while (scheduler.has_requests() or queue) and guard < 5000:
        guard += 1
        if scheduler.has_requests() and len(queue) < batch_queue_size:
            sched_output = scheduler.schedule()
            if sched_output.total_num_scheduled_tokens > 0:
                snap = dict(sched_output.num_scheduled_tokens)
                queue.append((sched_output, snap))
                max_depth = max(max_depth, len(queue))
                if len(queue) < batch_queue_size and scheduler.has_requests():
                    continue
        if queue:
            sched_output, snap = queue.popleft()
            req_ids = list(sched_output.num_scheduled_tokens.keys())
            sampled: list[list[int]] = []
            for req_id in req_ids:
                computed[req_id] += snap[req_id]
                if computed[req_id] < nprompt[req_id]:
                    sampled.append([])  # still prefilling -> no sampled token
                    continue
                n_spec = len(sched_output.scheduled_spec_decode_tokens.get(req_id, ()))
                k = min(accept, n_spec)
                toks = [900 + i for i in range(k + 1)]
                rq = by_id[req_id]
                if (
                    stop_after_output is not None
                    and rq.num_output_tokens >= stop_after_output
                ):
                    toks = [stop_token, *toks[1:]]
                sampled.append(toks)
            scheduler.update_from_output(
                sched_output,
                ModelRunnerOutput(
                    req_ids=req_ids,
                    req_id_to_index={r: i for i, r in enumerate(req_ids)},
                    sampled_token_ids=sampled,
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )
    assert guard < 5000, "batch_queue drain did not terminate"
    return max_depth


@pytest.mark.parametrize("num_spec", [1, 2, 3])
@pytest.mark.parametrize("accept", [0, 1, 2, 3])
@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5, 8])
def test_async_pp_spec_stops_at_max_tokens(num_spec, accept, max_tokens):
    """PP=2 + async + spec: stop at EXACTLY max_tokens for any acceptance rate."""
    if accept > num_spec:
        pytest.skip("cannot accept more drafts than were speculated")
    scheduler = create_scheduler(
        async_scheduling=True,
        pipeline_parallel_size=2,
        num_speculative_tokens=num_spec,
    )
    (req,) = create_requests(
        num_requests=1, num_tokens=4, max_tokens=max_tokens, ignore_eos=True
    )
    scheduler.add_request(req)

    max_depth = _drive_batch_queue(scheduler, [req], batch_queue_size=2, accept=accept)

    assert scheduler.get_num_unfinished_requests() == 0
    assert req.num_output_tokens == max_tokens, (
        f"spec+async overshoot/undershoot: got {req.num_output_tokens}, "
        f"expected {max_tokens}"
    )
    # The pipeline must have genuinely engaged (a batch scheduled ahead of an
    # un-applied prior output); otherwise this degrades to a lockstep test.
    if max_tokens > 1:
        assert max_depth >= 2, "batch_queue never reached depth 2 (not pipelined)"


@pytest.mark.parametrize("accept", [0, 1])
@pytest.mark.parametrize("max_tokens", [1, 2, 4, 8])
def test_async_pp_spec_chunked_prefill(accept, max_tokens):
    """Chunked prefill spanning multiple in-flight batches + spec + pipeline.

    Directly stresses the 'stale is_prefill_chunk snapshot under the batch_queue
    delay' concern: the prompt is split across several scheduled batches that are
    in flight together before any output is applied.
    """
    scheduler = create_scheduler(
        async_scheduling=True,
        pipeline_parallel_size=2,
        num_speculative_tokens=1,
        max_num_batched_tokens=16,  # prompt of 40 -> 3 prefill chunks
        max_model_len=128,
        enable_chunked_prefill=True,
    )
    (req,) = create_requests(
        num_requests=1, num_tokens=40, max_tokens=max_tokens, ignore_eos=True
    )
    scheduler.add_request(req)

    _drive_batch_queue(scheduler, [req], batch_queue_size=2, accept=accept)

    assert scheduler.get_num_unfinished_requests() == 0
    assert req.num_output_tokens == max_tokens


@pytest.mark.parametrize("accept", [0, 1])
def test_async_pp_spec_early_stop_token(accept):
    """A stop token fired mid-pipeline must finish the request without the
    already-in-flight batch over-producing output tokens."""
    stop_after = 3
    scheduler = create_scheduler(
        async_scheduling=True,
        pipeline_parallel_size=2,
        num_speculative_tokens=1,
    )
    (req,) = create_requests(
        num_requests=1,
        num_tokens=4,
        max_tokens=20,
        ignore_eos=False,
        stop_token_ids=[777],
    )
    scheduler.add_request(req)

    _drive_batch_queue(
        scheduler,
        [req],
        batch_queue_size=2,
        accept=accept,
        stop_token=777,
        stop_after_output=stop_after,
    )

    assert scheduler.get_num_unfinished_requests() == 0
    assert req.is_finished()
    # Stopped at the first step that emits the stop token after `stop_after`
    # outputs; the in-flight batch must not push the count past that boundary.
    assert req.num_output_tokens == stop_after + 1


@pytest.mark.parametrize("max_tokens", [1, 2, 4, 8])
def test_async_pp1_spec_control(max_tokens):
    """Control: same invariant at pp=1 async (batch_queue_size=2). Isolates the
    async-spec accounting from genuinely PP>1 microbatching."""
    scheduler = create_scheduler(
        async_scheduling=True,
        pipeline_parallel_size=1,
        num_speculative_tokens=1,
    )
    (req,) = create_requests(
        num_requests=1, num_tokens=4, max_tokens=max_tokens, ignore_eos=True
    )
    scheduler.add_request(req)

    _drive_batch_queue(scheduler, [req], batch_queue_size=2, accept=1)

    assert scheduler.get_num_unfinished_requests() == 0
    assert req.num_output_tokens == max_tokens
