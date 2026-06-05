# Foundations — what an LLM is made of, and why each part exists

> **Who this is for.** A strong software engineer who is *new to the ML internals*.
> You know what a token is, you've seen Markov chains and state machines, you can read
> matrix math — but you've consumed AI, not built it. This document gives you the
> mental model the rest of the docs assume. After it, `PIPELINE-NARRATIVE.md` (the
> systems story) and `PIPELINE.md` (the precise map) will click, because you'll know
> *what* an embedding / layer / head / KV cache **is** before you read *how* vLLM
> schedules and pages them.
>
> **The ladder.** `FOUNDATIONS.md` (this — what the model is) → `PIPELINE-NARRATIVE.md`
> (why the serving system is shaped this way) → `PIPELINE.md` (where everything lives,
> `file:line`) → `SPEC-PP-INVARIANTS.md` (what you must not break when you change it).
>
> **How to read.** Narrative, top to bottom, with examples. Math is kept intuitive but
> honest — where a formula earns its place, it's here, explained term by term. Code
> anchors (`file:line`) are included so you can immediately see the bridge from concept
> to the vLLM tree, but you don't need to open them on the first read.

---

## 0. Start from what you already know: a Markov chain

You know a Markov chain: you're in a state, and a fixed table of probabilities tells
you the next state. "The weather tomorrow depends only on the weather today."

An LLM is the same shape with three upgrades:

1. **The "state" is the entire text so far**, as a sequence of tokens — not just the
   last symbol. (So it's really a very-high-order Markov chain.)
2. **The transition probabilities are not stored in a table — they are *computed*** by
   a big learned function (the neural network). A table for "all possible texts" would
   be infinite; the network compresses that into ~billions of numbers (weights).
3. **The output is a probability distribution over the next *token***, from which you
   pick one, append it, and repeat.

That loop is the whole game:

```
tokens so far  ─▶  model  ─▶  probability over the next token  ─▶  pick one  ─▶  append
       ▲                                                                          │
       └──────────────────────────────────────────────────────────────────────────┘
```

This is called **autoregression** ("regressing on your own past output"). Everything
else in this document — embeddings, attention, KV cache, the head — is just *what's
inside that "model" box*, and everything in the vLLM docs is about doing this loop for
**thousands of sequences at once, fast, on scarce GPU memory, without corrupting the
output.**

Hold onto one number: a typical model runs this loop to produce **one token per pass**.
That single fact drives almost every design decision later (it's why decode is slow, why
KV caching exists, why speculative decoding is worth it).

---

## 1. Tokens → embeddings: turning symbols into geometry

**Tokenization.** Text is first chopped into **tokens** — subword chunks, each with an
integer id. `"unhappiness"` might become `["un", "happ", "iness"]` → `[1023, 5519,
8841]`. The vocabulary is fixed (e.g. 250,000 tokens for Qwen3.5). You can think of a
token id as "a row number."

**Why we can't feed the ids directly.** The id `5519` is just an index — it carries no
meaning. `5519` is not "bigger" or "closer to" `5520` in any useful sense. The network
needs *meaning expressed as geometry*: similar tokens should be nearby points in space,
so that math (dot products, sums) can operate on meaning.

**The embedding is a lookup table.** It's a matrix of shape `[vocab_size, hidden_size]`
— one row per token, each row a vector of `hidden_size` floats (e.g. 5120). "Embedding
token 5519" literally means "fetch row 5519." Those rows are *learned* during training
so that, e.g., the vectors for "king" and "queen" end up related.

```
token id 5519  ──lookup──▶  [0.21, -1.03, 0.44, … , 0.07]   (a 5120-dim vector)
                            └──────── this token's "meaning vector" ────────┘
```

**Why this matters for vLLM (the bridge).** That table is huge: `250k × 5120 × 2 bytes
≈ 2.4 GB` just for the embedding. In vLLM it's `VocabParallelEmbedding`
(`vllm/model_executor/layers/vocab_parallel_embedding.py:192`). When you read in the
project notes that "A1c quantizes the *draft's* embedding to int4," now you know exactly
what's being shrunk and why it's worth ~1.8 GB — it's this table, duplicated for the
speculative draft model. (And quantizing it is *safe* — we'll see why in §7.)

The output of this stage is the sequence turned into a stack of vectors:
`[num_tokens, hidden_size]`. This block of numbers is called the **hidden states** — the
model's working representation of the text. It will be transformed, layer by layer.

---

## 2. The layers (transformer blocks): where "thinking" happens

The model is a **stack of identical blocks** (e.g. 64 of them). Each block takes the
hidden states in and produces refined hidden states out, same shape. Stacking them lets
the model build up increasingly abstract representations — early layers ≈ syntax, late
layers ≈ meaning/task (roughly).

Each block does **two** things:

### 2a. Attention — mix information *across* tokens

This is the heart of the transformer, and the one concept worth slowing down on.

Up to now each token's vector only knows about *itself*. But "it" in "the cat sat
because **it** was tired" needs to look back at "cat." **Attention** is the mechanism
for a token to *look at the previous tokens and pull in a weighted blend of their
information.*

For each token, the block computes three vectors from its hidden state, each via a
learned matrix:
- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I offer, as something to be looked at?"
- **Value (V)** — "the actual information I'll hand over if attended to."

Then, for token *i*:
1. Compare its **Q** to every previous token's **K** (dot product) → a relevance score
   per past token. High score = "that token is relevant to me."
2. Softmax those scores into weights that sum to 1.
3. Output = the weighted sum of the past tokens' **V** vectors.

In one line (the famous formula), for the whole sequence at once:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```

- `Q·Kᵀ` — every token's query dotted with every token's key → an `[n × n]` grid of
  relevance scores.
- `/ √d` — scale so the softmax doesn't saturate (d = head dimension). Housekeeping.
- `softmax(...)` — turn each row of scores into weights summing to 1.
- `· V` — blend the values by those weights.

**Causal masking.** A token may only look *backward* (you can't attend to words you
haven't generated yet). So the upper triangle of that `[n × n]` grid is masked to zero.
This is why it's "causal self-attention."

**Heads.** This is done in parallel by several **attention heads**, each with its own
Q/K/V matrices, so different heads can specialize ("one tracks subjects, one tracks
positions"). Their outputs are concatenated. That's the "multi-head" in "multi-head
attention."

### 2b. MLP / feed-forward — transform *each* token

After attention has mixed information across tokens, a small two-layer neural net
(the **MLP** or **feed-forward** block) is applied to *each token's vector
independently* — "now that you've gathered context, refine yourself." This is where a
large share of the parameters live.

### 2c. The plumbing that makes deep stacks trainable

- **Residual connections**: each block adds its output to its input (`x + block(x)`)
  rather than replacing it — so information and gradients flow through 64 layers
  without vanishing. (This is the `residual` you'll see travel between PP stages.)
- **Normalization** (RMSNorm/LayerNorm): rescales vectors to keep numbers stable.

**The bridge.** In vLLM the per-token `{hidden_states, residual}` pair is exactly what
crosses a pipeline-parallel stage boundary (`vllm/sequence.py:12`, `IntermediateTensors`)
— now you know it's just "the working representation, mid-stack, handed to the GPU that
holds the next layers."

---

## 3. The KV cache — the single most important systems idea, and *why*

This is the concept that explains 80% of the vLLM memory machinery. Get this and the
rest of the docs open up.

**The problem.** To generate token N+1, attention (§2a) needs the **K** and **V** of
*every* previous token, at *every* layer. If you recomputed K and V for the whole prefix
on every single step, generating a 1000-token answer would cost ~1000× the attention
work — O(n²) total. Painfully slow.

**The insight.** The K and V of a past token **never change** once computed. Token 5's
key is token 5's key forever. So **compute them once and cache them.** On each new step,
the new token computes only *its own* Q, K, V, appends its K/V to the cache, and attends
its Q against the *entire cached* K/V.

```
step N:   new token computes Q,K,V  →  append K,V to cache  →  attend Q over ALL cached K,V
                                          (cache grows by 1 token × every layer)
```

So decode becomes **O(1) recompute per step** — but at a price: you must **store** K and
V for every token × every layer × every attention head. That store is the **KV cache**,
and for long sequences it **dominates GPU memory** — more than the model weights.

**Now every vLLM memory decision is obvious:**
- The KV cache is huge and per-request → you can't give each request a fixed max-size
  buffer (waste + fragmentation) → **PagedAttention** stores it in fixed **blocks** with
  a per-request **block table**, like OS virtual memory. (`PIPELINE.md` §7.)
- Blocks can run out mid-step → the scheduler must **preempt** requests.
- Identical prefixes (same system prompt) can **share** blocks → **prefix caching**.
- The speculative draft has its *own* K/V (its own attention layer) but shares the block
  tables — that's brick 30's whole topic.

**A number to feel it:** for a 7B model at a few thousand tokens of context, the KV cache
is often several GB *per* concurrent request. On a 16 GB GPU, "how many KV blocks fit"
*is* "how many requests can I serve at once." That's why the worker runs a profiling pass
to size `num_gpu_blocks` (`PIPELINE.md` §7) before anything else.

---

## 4. The head and logits — turning a vector back into a token

After the last block, each position has a final hidden-state vector — the model's
fully-cooked representation of "what comes next here." To turn that back into an actual
token you need one more step.

**The LM head** is a matrix of shape `[hidden_size, vocab_size]`. Multiplying the final
hidden vector by it produces one score per vocabulary token — the **logits**:

```
final hidden vector  ──LM head ([hidden × vocab])──▶  logits = [score per token, ×250k]
```

- **Logits** are raw, unbounded scores. `softmax(logits)` turns them into a probability
  distribution over the vocabulary.
- **Sampling** picks the actual next token:
  - **Greedy / argmax** — take the single highest-logit token. Deterministic.
  - **Temperature / top-p / top-k** — flatten or restrict the distribution and sample
    randomly, for diversity.

**Two bridges you'll need later:**
1. The LM head is often **tied** to the embedding table (same weights, transposed) — one
   matrix does "id → vector" on the way in and "vector → scores" on the way out. (This is
   why the project notes say the draft "shares the target's lm_head.")
2. `argmax` being deterministic is *the* hook behind speculative decoding's correctness
   (§7): greedy output is a fixed function of the logits, so a draft token is "right"
   iff it equals the target's argmax.

In vLLM this is `compute_logits` + the sampler; the head lives only on the **last**
pipeline stage (the embedding only on the **first**) — which, now that you know what each
does, is the natural place for them.

---

## 5. One full pass, end to end

Putting §1–§4 together, here is a single forward pass producing a single token:

```
token ids
   │  §1 embedding lookup            [n, hidden]
   ▼
 ┌─────────────────────────────────────────────┐
 │  block 1:  attention (over KV cache) + MLP   │  §2
 │  block 2:  …                                 │
 │   …  (×64)                                    │
 └─────────────────────────────────────────────┘
   │  final norm                     [n, hidden]
   ▼
   §4 LM head → logits [n, vocab]  →  take the LAST position's logits
   ▼
   sample (argmax / temperature)  →  next token id
   ▼
   append to the sequence, write this step's K/V into the cache, and loop  (§0, §3)
```

That's a complete language model. **Prefill** = run this over the whole prompt at once
(fills the KV cache for all prompt tokens, produces the first new token). **Decode** =
run it one token at a time afterward (each step adds one K/V to the cache). vLLM's
scheduler deliberately *erases* the prefill/decode distinction — it just tracks "how many
of this request's tokens have been computed" — which is the "no phases" idea in
`PIPELINE-NARRATIVE.md` §Move 1/3.

**Why decode is slow (and why we care).** In decode you stream *all* the weights and the
*entire* KV cache through the GPU to produce *one* token. The GPU's compute units sit
mostly idle waiting on memory — decode is **memory-bandwidth-bound**, not compute-bound.
That single fact is the whole motivation for the next two sections.

---

## 6. Pipeline parallelism, in one paragraph (you already have it)

The model (§2) is a stack of layers. If it doesn't fit on one GPU, **split the stack**:
GPU0 holds layers 0–k (and the embedding, §1), GPU1 holds layers k–end (and the head,
§4). A token's hidden state flows GPU0 → GPU1 once per boundary (the `IntermediateTensors`
from §2c). That's **pipeline parallelism (PP)**. The alternative, tensor parallelism,
splits *every* layer and must synchronize *every* layer — fine over fast NVLink, fatal
over slow PCIe. Full story: `PIPELINE-NARRATIVE.md` §Move 4. The only thing to carry
forward: under PP, the embedding is on the first GPU, the head on the last, and the
speculative draft (next section) naturally lives on the last GPU because that's where the
final hidden state already is.

---

## 7. Speculative decoding & MTP — the project, finally in context

Now the part the whole project is about.

**The opportunity.** Decode is memory-bound (§5): producing one token barely uses the
GPU's compute. So — *could we check several candidate tokens for almost the same cost as
one?* If we already had a guess for the next few tokens, the target model could verify
them all in a **single** forward pass (attention over the same KV cache, just a few more
positions — nearly free on a memory-bound GPU).

**The catch.** You can't know token N+1 before producing token N — that's autoregression
(§0). So you need a **cheap guesser** (a "draft") and a way to **verify** its guesses
without changing the output.

**The drafter.** Several kinds exist (n-gram, EAGLE, draft-model), but ours is **MTP =
Multi-Token Prediction**. A normal model predicts *one* next token from the final hidden
state (§4). MTP bolts on a small extra module — its own little embedding + one extra
transformer layer + the *shared* head — that, given the **same final hidden state the
target just produced**, predicts the token *after* next. Because:
- its input (the target's hidden state) is already computed → **free**,
- it's tiny (one layer) → **cheap**,
- it was trained to imitate the target → its guesses are often right (~71% accepted on
  llama.cpp),

it's an excellent draft generator. In vLLM it's a model like
`vllm/model_executor/models/qwen3_5_mtp.py`. Now you can read brick 20 ("the draft has
its own embedding") and brick 30 ("the draft reuses the target's hidden state, resident
on the last rank") and they mean something concrete.

**The verification (why it's provably correct).** The **rejection sampler** is the
oracle. In greedy mode the rule is exactly:

> a draft token is **accepted** iff it equals the target model's `argmax` at that
> position; otherwise it is **replaced** by the target's argmax and all later drafts are
> thrown away.

Walk through why that's identical to normal greedy decoding: normal greedy *would have*
emitted the target's argmax at each position anyway. If the draft guessed it, great — we
skipped recomputing it. If the draft guessed wrong, we substitute the correct argmax — so
the emitted token is *still* exactly what greedy would have produced. The draft only ever
saves work; it can never change the answer. (`vllm/v1/sample/rejection_sampler.py:708`
and `:452`.)

**The two consequences that drive the project:**
1. **It's weight-agnostic.** Since correctness depends only on (target argmax, draft
   guess), the draft's *quality* changes only how often you accept — never the output.
   This is the license to make the draft **lossy**: quantizing the draft's 2.4 GB
   embedding to int4 (A1c) can only lower acceptance, never corrupt text. It's also why
   dummy/MiMo weights are valid for *testing* the machinery.
2. **The bonus token & the grid.** If *all* K drafts are accepted, the target also gives
   one extra "free" token (it computed those logits anyway). So the sampler's output is a
   grid `[num_reqs, num_spec+1]`, valid tokens packed from the left, `-1` padding on the
   right. That `-1` and that variable width are exactly the things that have to survive
   the pipeline correctly — the bugs the project is fixing (`PIPELINE-NARRATIVE.md`
   §Move 7).

**Why "spec × PP × async" is hard (the one-sentence version you can now parse):** the
draft is produced on the *last* GPU, but the *earlier* GPUs need the resulting tokens to
build the next step's input; and because the engine pipelines `pp_size` steps ahead, the
per-step draft bookkeeping (those `-1` placeholders) has to stay correct across a
multi-step delay. That's the seam.

---

## 8. The compact math primer

Just enough to follow papers and reason out loud — you already have the linear algebra.

- **Everything is a matmul.** Hidden states are a matrix `[n_tokens, hidden]`. A linear
  layer is `X·W`. Attention and MLP are stacks of these. "The model is N billion
  parameters" = the total size of all the `W` matrices.
- **logits → probabilities.** `softmax(z)_i = e^{z_i} / Σ_j e^{z_j}` — exponentiate and
  normalize so scores become a distribution. `argmax` picks the largest; **temperature
  T** divides logits by T before softmax (T<1 sharpens → more deterministic, T>1
  flattens → more random); **top-p** keeps only the smallest set of tokens whose
  probabilities sum to p.
- **attention** `softmax(QKᵀ/√d)·V` — §2a, term by term. The `QKᵀ` is `[n, n]` (every
  token vs every token), which is why long contexts are expensive and why the KV cache
  (caching K, V) is the key optimization.
- **compute-bound vs memory-bound.** Prefill processes many tokens at once → lots of
  matmul per byte loaded → **compute-bound**. Decode processes one token but must reload
  all weights + KV → little matmul per byte → **memory-bound**. Speculative decoding
  exploits exactly this: extra candidate tokens are nearly free on a memory-bound step.
- **why rejection sampling is exact** (the non-greedy case, for completeness): to sample
  from the target distribution `p` while proposing from draft `q`, accept the draft token
  `x` with probability `min(1, p(x)/q(x))`, and on rejection sample from the normalized
  positive part of `p−q`. The math guarantees the final samples are distributed exactly
  as if drawn from `p`. Greedy (§7) is the degenerate case where `p` is a spike on the
  argmax, which collapses to "accept iff equal." (Paper: Leviathan et al. /
  Chen et al., 2022–2023.)

---

## 9. Concept → vLLM map (your bridge to the other docs)

| Concept (here) | What it is | In vLLM (`file:line`) |
|---|---|---|
| Token embedding | id → meaning vector (lookup table) | `vocab_parallel_embedding.py:192` (`VocabParallelEmbedding`) |
| Hidden states | the working representation, `[n, hidden]` | flows as `IntermediateTensors`, `sequence.py:12` |
| Transformer block | attention + MLP + residual/norm | model files, e.g. `qwen3_5_mtp.py` |
| KV cache | cached K/V to avoid O(n²) recompute | paged: `block_table.py:70`, `PIPELINE.md` §7 |
| LM head / logits | final vector → per-token scores | `compute_logits` / `lm_head` (last PP rank) |
| Sampling / argmax | logits → next token | sampler; greedy = argmax |
| Pipeline parallelism | split the layer stack across GPUs | `PIPELINE-NARRATIVE.md` §Move 4, brick 10 |
| MTP draft | tiny module reusing the target's hidden state | `qwen3_5_mtp.py`, bricks 20/30 |
| Rejection sampler | the correctness oracle (argmax-match) | `rejection_sampler.py:708`, brick 80 §5 |
| `-1` placeholder grid | `[num_reqs, num_spec+1]`, `-1`-padded | `rejection_sampler.py:30`, `PIPELINE.md` §8 |

**Next on the ladder:** read `PIPELINE-NARRATIVE.md` — every "forced move" there now
rests on a concept you own. Then, when you go to change the spec path, `SPEC-PP-INVARIANTS.md`
will tell you what you must not break.
