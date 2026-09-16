---
layout: post
title: "Transformer Inference Arithmetic — A Worked Walkthrough"
subtitle: "KV-cache memory, batching, tensor parallelism, communication, and latency—derived with one 52B example."
date: 2026-09-16
categories: [Tech, machine-learning, llm]
tags: [machine-learning, llm, transformers, inference, gpu]
reading_time: 28
description: "A first-principles walkthrough of Transformer inference arithmetic: KV-cache size, weight bandwidth, batching, tensor parallelism, communication, FLOPs, and real-hardware corrections."
featured: true
---

_My notes on [kipply's "Transformer Inference Arithmetic"](https://kipp.ly/p/transformer-inference-arithmetic), rebuilt from first principles with one consistent example, full derivations, and no magic numbers._

**Want the short version? [Jump straight to the TL;DR.](#tldr)**

---

## Why this matters

If you serve an LLM, these calculations answer practical questions:

- How many requests can fit in one batch?
- What does one more token of context cost?
- Is this step limited by math, HBM bandwidth, or GPU-to-GPU communication?
- Will adding another GPU reduce latency, or only add communication?

The goal is not to predict latency to the microsecond. It is to build a model simple enough to calculate by hand and accurate enough to identify the right bottleneck.

## Our running example

| Quantity                                   |                   Value |
| ------------------------------------------ | ----------------------: |
| Model size                                 |          52B parameters |
| `d_model`                                  |                   8,192 |
| `n_layers`                                 |                      64 |
| `n_heads × d_head`                         |      `64 × 128 = 8,192` |
| `vocab_size`                               |                  50,257 |
| Weight/cache format                        | BF16, 2 bytes per value |
| GPU                                        |        NVIDIA A100 40GB |
| Peak BF16 compute                          |             312 TFLOP/s |
| HBM bandwidth                              |                1.5 TB/s |
| Effective one-direction GPU link bandwidth |                300 GB/s |

This is the same illustrative 52B shape used in kipply's post. It assumes ordinary multi-head attention and a two-matrix MLP with a `4d` hidden width. Modern models may use grouped-query attention (GQA), gated MLPs, RoPE, quantized weights, or different accelerators. Those change constants, not the method.

> **Important:** hardware peak numbers produce **lower bounds**, not promises. Unless stated otherwise, GB and TB below are decimal units, while "MiB" and "GiB" are binary units.

---

## Part 1 — The architecture, in matrix shapes

### 1.1 Tokens become residual-stream vectors

A tokenizer turns text into integer token IDs. An embedding table

$$E \in \mathbb{R}^{V \times d}$$

maps each ID to a vector of width `d = d_model`. Positional information is then added or applied. Older models may use learned position embeddings; many modern models rotate Q and K with RoPE instead.

For a sequence of length `S`, the residual stream has shape

$$X \in \mathbb{R}^{S \times d} = \mathbb{R}^{S \times 8192}.$$

Every Transformer block accepts and returns that same shape. Only the vocabulary head changes the final dimension from `d` to `V`.

![Decoder-only Transformer: tokenization and embedding, repeated attention and MLP blocks, then a vocabulary head](/assets/img/transformer-param-count/01-architecture.svg)

### 1.2 What self-attention computes

For one layer,

$$Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V,$$

with

$$W_Q,W_K,W_V \in \mathbb{R}^{d \times d}.$$

After splitting the result into heads, one head computes

$$A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}} + M_{\text{causal}}\right),\qquad Z = AV.$$

The distinction is essential:

- `q · k` produces a **score**, not a value.
- The softmax turns all scores into weights.
- Those weights form a weighted sum of the `v` vectors.

`K` answers "how relevant is this position to my query?" `V` supplies the content that position contributes.

For a full `S`-token prefill, each head's score matrix is `S × S`. During cached decoding, the new query has shape `1 × d_head`, so it produces only a `1 × t` row against the `t` cached keys.

### 1.3 Why cache K and V, but not Q?

Causal masking creates the asymmetry:

- A token's query is consumed while computing **that token's** output. Future tokens bring their own queries.
- A token's key and value are read by **every later token** that attends to it.
- Once computed at a layer, that token's K and V cannot be changed by future tokens.

<img src="/assets/img/transformer-inference-arithmetic/01-kv-cache.svg" alt="A token's query is used once and discarded, while its key and value persist in the cache and are reused by future queries." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

The cache stores answers already computed. It does not change the model's result; it avoids repeating the same K/V projections.

### 1.4 What one block contains

Under this post's assumptions, one block has six large matrices:

- attention: `Wq`, `Wk`, `Wv`, `Wo`, each `d × d`;
- MLP: `W1`, shape `d × 4d`, and `W2`, shape `4d × d`.

LayerNorm, residual additions, the activation, masking, and softmax are also real operations. They contain far fewer parameters and FLOPs, but their data movement is not always free. We return to that in Part 9.

---

## Part 2 — The KV cache, quantified

### 2.1 Bytes stored per token

For ordinary multi-head attention, each token stores one K vector and one V vector at every layer:

$$
\begin{aligned}
\text{KV bytes/token}
&= 2_{\text{K,V}}
   \cdot b_{\text{bytes/value}}
   \cdot n_{\text{layers}}
   \cdot n_{\text{heads}}
   \cdot d_{\text{head}} \\
&= 2 \cdot 2 \cdot 64 \cdot 64 \cdot 128 \\
&= 2{,}097{,}152\text{ bytes} \\
&= 2\text{ MiB}.
\end{aligned}
$$

Because `n_heads · d_head = d_model`, the BF16 formula simplifies to

$$\boxed{\text{KV bytes/token} = 4\,n_{\text{layers}}d_{\text{model}}.}$$

That means one 2,048-token request needs

$$2\text{ MiB/token} \times 2048 = 4\text{ GiB}$$

of KV cache across the tensor-parallel group.

For GQA or multi-query attention, replace `n_heads` with the smaller `n_kv_heads`:

$$\text{KV bytes/token} = 2b\,n_{\text{layers}}n_{\text{kv-heads}}d_{\text{head}}.$$

This is one reason serving-oriented models use fewer KV heads.

### 2.2 Projection FLOPs saved by caching

Projecting one token through K and V at all layers costs

$$
2_{\text{matmul}}
\cdot 2_{\text{K,V}}
\cdot n_{\text{layers}}
\cdot d^2
= 4 \cdot 64 \cdot 8192^2
= 17{,}179{,}869{,}184
$$

or about **17.18 GFLOPs per old token**.

The six large matrices in a complete block stack cost about 103.08 GFLOPs per token (derived in Part 8), so K and V projections are exactly one-sixth of that dense-matmul baseline.

At decode position `t`, the cache avoids repeating those K/V projections for the previous `t` positions. It still:

1. computes K and V for the **new** token;
2. reads old K/V from the cache;
3. computes the new query against all cached keys;
4. forms a weighted sum of all cached values.

### 2.3 A complexity correction worth remembering

It is tempting to say that a KV cache changes generation from quadratic to linear. That is not quite right.

- Without a cache, each decode step reruns projections over the whole prefix.
- With a cache, dense projection/MLP work is done only for the new token.
- But the new query still scans `t` cached keys and values.

So the **attention scan per decode step remains `O(t)`**, and dense autoregressive attention over a whole generated sequence remains `O(L²)` in sequence length. The cache removes repeated prefix computation; it does not make attention itself constant-time.

In practice, for a very wide model and moderate context, streaming the dense weights can dominate enough that consecutive decode steps look almost flat. At long contexts, KV-cache reads and attention become visible.

---

## Part 3 — Where compute actually happens

There are two different "load the model" events:

1. **Storage → GPU HBM:** load the checkpoint at process startup.
2. **HBM → on-chip SRAM/registers/tensor cores:** stream weight tiles during every forward pass.

The checkpoint remains resident in HBM, but 104GB cannot remain in the GPU's tiny on-chip memories. Kernels repeatedly stage tiles from HBM, multiply them, and make room for the next tiles.

For 52B BF16 parameters:

$$52\text{B} \times 2\text{ bytes} = 104\text{ GB of weights}.$$

On an imaginary single A100 large enough to hold them, the ideal time to read those weights once would be

$$T_{\text{weights,1}} = \frac{104\text{e}9}{1.5\text{e}12} = 69.3\text{ ms}.$$

The real model does not fit on one 40GB A100. With four-way tensor parallelism, each GPU reads roughly 26GB:

$$T_{\text{weights,4}} = \frac{26\text{e}9}{1.5\text{e}12} = 17.3\text{ ms}.$$

### 3.1 Why batching amortizes weight traffic

Suppose `B` token positions use the same matrix in one matrix-matrix multiplication. The weights are read once, while the math scales with `B`:

$$
\text{arithmetic intensity} \approx
\frac{2BP\text{ FLOPs}}{2P\text{ bytes}}
= B\text{ FLOPs/byte}.
$$

The A100 hardware balance is

$$\frac{312\text{e}12\text{ FLOP/s}}{1.5\text{e}12\text{ byte/s}} \approx 208\text{ FLOPs/byte}.$$

Therefore the ideal dense-matmul ridge point is near **`B = 208`**:

- below 208, weight bandwidth is the lower-bound bottleneck;
- above 208, peak math throughput is the lower-bound bottleneck.

For decode, `B` usually means concurrent sequences contributing one new token each. Prefill also processes many token positions together, although causal attention and exact matrix shapes make it less identical than this simple model suggests.

<img src="/assets/img/transformer-inference-arithmetic/02-batching-crossover.svg" alt="Ideal batching roofline for the 52B model on four A100 GPUs: a 17.3 millisecond weight-streaming floor meets the compute line near a batch of 208." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

Using only the six large matrices:

| Positions processed together (`B`) | Weight floor, TP=4 | Ideal math time, TP=4 | Dense-matmul lower bound |
| ---------------------------------: | -----------------: | --------------------: | -----------------------: |
|                                  1 |            17.3 ms |              0.083 ms |                  17.3 ms |
|                                 50 |            17.3 ms |               4.13 ms |                  17.3 ms |
|                                100 |            17.3 ms |               8.26 ms |                  17.3 ms |
|                            **208** |        **17.3 ms** |           **17.2 ms** |             **≈17.3 ms** |
|                                500 |            17.3 ms |               41.3 ms |                  41.3 ms |

This does **not** mean all extra tokens are literally free below 208. It means the idealized large matmuls can reuse weights while math hardware that would otherwise wait on memory does useful work. KV reads, attention, activation traffic, communication, and kernel overhead give the memory line a slope in reality.

---

## Part 4 — Capacity: does it fit?

Weights alone require

$$\left\lceil\frac{104}{40}\right\rceil = 3\text{ A100-40GB GPUs}.$$

The remaining aggregate HBM is the first approximation of the KV-cache budget:

| GPUs | Total HBM | Weight memory | Theoretical remainder | Theoretical KV-token capacity |
| ---: | --------: | ------------: | --------------------: | ----------------------------: |
|    3 |    120 GB |        104 GB |                 16 GB |                 ≈7,629 tokens |
|    4 |    160 GB |        104 GB |                 56 GB |                ≈26,703 tokens |

The final column is simply `remainder / 2,097,152 bytes per token`.

These are **ceilings**, not safe scheduler limits. A serving runtime also needs:

- temporary activations and workspaces;
- communication buffers;
- allocator headroom and memory lost to fragmentation;
- CUDA context and kernel-library allocations;
- possibly untied vocabulary-head weights.

Consequently, a practical four-GPU limit can be noticeably below 26,700 cached tokens. This is not a rounding issue; it is reserved and fragmented memory.

The scheduling consequence is still clear: after the weights fit, extra HBM can increase concurrency, and concurrency lets the server build larger efficient batches.

---

## Part 5 — Splitting weights across GPUs

Tensor parallelism shards each large matrix. With four GPUs, each package owns roughly one quarter of the weight bytes and performs roughly one quarter of the matmul work.

<img src="/assets/img/transformer-inference-arithmetic/03-tensor-parallel.svg" alt="Four tensor-parallel GPUs each read a 26GB weight shard, compute a partial output, and exchange partials in a collective to reconstruct the full activation." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

The GPUs are not merely a shared pool of memory. Each GPU has its own HBM controllers and compute units. Four GPUs therefore provide four independent weight streams and four sets of tensor cores.

### 5.1 Why communication is required

Depending on how a matrix is partitioned, each GPU may produce:

- a distinct slice that can be concatenated; or
- a partial sum that must be reduced across GPUs.

A common Megatron-style Transformer block arranges the sharding so that it needs **two logical activation all-reduces per block**:

1. after attention's output projection;
2. after the MLP's down-projection.

An all-reduce is itself implemented as multiple network phases—for example, reduce-scatter plus all-gather. This is why some descriptions count four communication phases per block. Calling all four phases "four all-reduces" would overcount the logical synchronization points.

### 5.2 A simple communication-volume model

One BF16 activation payload for `B` positions is

$$M_{\text{payload}} = 2Bd\text{ bytes}.$$

For `B=500` and `d=8192`,

$$M_{\text{payload}} = 2 \cdot 500 \cdot 8192 = 8.192\text{ MB}.$$

For a ring all-reduce over `N` GPUs, a common per-GPU send-plus-receive estimate is

$$M_{\text{ring}} \approx 2\frac{N-1}{N}M_{\text{payload}}.$$

At `N=4`, that is about **12.288 MB per logical all-reduce**. Two reductions per layer across 64 layers move about 1.57GB per GPU:

$$12.288\text{ MB} \times 2 \times 64 \approx 1.57\text{ GB}.$$

At an idealized effective 300GB/s, the bandwidth term is

$$T_{\text{comm,volume}} \approx \frac{1.57\text{e}9}{300\text{e}9} = 5.24\text{ ms}.$$

The exact number depends on topology, collective algorithm, whether bandwidth is quoted per direction or aggregate, message size, and overlap.

---

## Part 6 — A latency model that does not contradict itself

A useful decomposition is

$$
T_{\text{step}}
\approx T_{\text{dense roofline}}
+ T_{\text{attention/cache}}
+ T_{\text{communication on critical path}}
+ T_{\text{fixed overhead}}.
$$

For the dense matrices,

$$
T_{\text{dense roofline}}
\approx
\max\left(
\frac{\text{weight bytes}}{N\cdot\text{HBM bandwidth}},
\frac{\text{dense FLOPs}\cdot B}{N\cdot\text{peak FLOP/s}}
\right).
$$

Communication may overlap with independent compute, but collectives also sit between dependent stages. Therefore:

$$
\max(T_{\text{compute}},T_{\text{comm}})
\leq T_{\text{compute+comm}}
\leq T_{\text{compute}}+T_{\text{comm}}.
$$

Neither "always add them" nor "always take the maximum" is universally correct.

### 6.1 Small decode batch

For `B=1`, TP=4:

- ideal weight-streaming floor: **17.3ms**;
- ideal math time: **0.083ms**;
- collective startup: if one logical collective costs an optimistic `8μs`, then

$$2 \cdot 64 \cdot 8\mu\text{s} \approx 1.0\text{ ms}.$$

That gives an illustrative **≈18.3ms** lower bound before long-context cache reads and runtime overhead. The `8μs` figure is only a toy latency assumption; real collective latency depends strongly on generation, topology, and implementation.

### 6.2 Large batch: `B=500`

The dense math term is

$$
\frac{103.08\text{e}9 \cdot 500}{4 \cdot 312\text{e}12}
\approx 41.3\text{ ms}.
$$

The ring-volume estimate from Part 5 is about **5.24ms**. A conservative serialized estimate is therefore

$$41.3 + 5.24 \approx 46.5\text{ ms},$$

plus attention, cache, and software overhead. With useful overlap it could be closer to 41.3ms; with inefficient collectives it could be higher than 46.5ms.

The throughput lesson survives either way. Producing 500 next-token positions in roughly tens of milliseconds is dramatically more efficient than running 500 separate weight-streaming passes. Batching improves aggregate throughput; it does not remove the sequential dependence between consecutive tokens of the **same** request.

---

## Part 7 — When does communication become the bottleneck?

The A100's compute-to-link ratio is

$$\frac{312\text{e}12\text{ FLOP/s}}{300\text{e}9\text{ byte/s}} \approx 1040\text{ FLOPs/byte}.$$

This is a second roofline. A stage that performs fewer than about 1040 useful FLOPs per communicated byte is at risk of becoming communication-bound.

For the output projection and MLP down-projection, a useful worst-case rule of thumb is that the per-GPU arithmetic intensity relative to an activation-sized collective scales like

$$\frac{d_{\text{model}}}{N}.$$

For our model:

- TP=4: `8192 / 4 = 2048 FLOPs/byte` — above 1040;
- TP=8: `8192 / 8 = 1024 FLOPs/byte` — approximately at the hardware balance;
- TP=16: `8192 / 16 = 512 FLOPs/byte` — communication is increasingly exposed.

This is not a universal cutoff. Faster links, topology-aware collectives, overlap, quantized communication, and different matrix shapes move it. But it explains the tradeoff:

> More GPUs reduce each GPU's weight traffic and math, while making the fixed-size activation exchange large relative to each GPU's shrinking share of work.

The minimum GPU count is a capacity question. The best GPU count is a latency-throughput-cost question.

---

## Part 8 — Where "FLOPs per token ≈ 2 × parameters" comes from

### 8.1 The one matmul rule

Multiplying an `m × n` matrix by a length-`n` vector performs approximately

$$2mn\text{ FLOPs},$$

counting one multiply and one add per matrix element. The matrix itself contains `mn` parameters. Therefore, if a weight is used once in a matmul,

$$\text{FLOPs} \approx 2 \times \text{parameters}.$$

### 8.2 The six large matrices in one block

| Matrix | Shape          |  Parameters | FLOPs/token |
| ------ | -------------- | ----------: | ----------: |
| `Wq`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wk`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wv`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wo`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `W1`   | `8192 × 32768` | 268,435,456 | 536,870,912 |
| `W2`   | `32768 × 8192` | 268,435,456 | 536,870,912 |

One block therefore costs

$$
4(2d^2) + 2(2d\cdot4d)
= 24d^2
= 1{,}610{,}612{,}736\text{ FLOPs/token}.
$$

Across 64 layers:

$$
24 \cdot 64 \cdot 8192^2
= 103{,}079{,}215{,}104
\approx 103.08\text{ GFLOPs/token}.
$$

Dividing by two recovers the block-weight count:

$$\frac{103.08\text{B}}{2} = 51.54\text{B parameters}.$$

### 8.3 What the `2P` shortcut leaves out

The 103.08 GFLOPs figure is the dense **projection + MLP baseline**, not the whole decode step.

**Cached attention.** At context length `t`, QK scores and the weighted-V operation add roughly

$$4t\,d\,n_{\text{layers}}\text{ FLOPs}.$$

At `t=2048`, that is

$$
4 \cdot 2048 \cdot 8192 \cdot 64
= 4.29\text{ GFLOPs},
$$

about 4.2% of the six-matrix baseline. At longer contexts it grows linearly per decode step.

**Vocabulary head.** Projecting the final hidden state to 50,257 logits costs

$$
2Vd
= 2 \cdot 50{,}257 \cdot 8192
\approx 0.823\text{ GFLOPs}.
$$

The embedding table contains about 411.7M parameters. If embeddings are tied, those same parameters act as the vocabulary head and **do** participate in a matmul at output time, even though input embedding is only a lookup. If the head is untied, the model stores another matrix.

**Other kernels.** LayerNorm, RoPE, residuals, activation functions, softmax, sampling, and cache reads add smaller FLOP counts but nonzero latency.

So `2P` is powerful because most large-model parameters are used once per token in dense matmuls. It is not an exact law.

---

## Part 9 — Memory traffic the FLOP count misses

FLOP accounting makes LayerNorm, softmax, residual additions, and activations look trivial. On a GPU, an operation can do very little arithmetic and still spend time reading and writing tensors.

Three effects matter:

1. **Intermediate activation traffic.** Unfused kernels may write a tensor to HBM only for the next kernel to read it back.
2. **KV-cache traffic.** Each decode step reads all earlier K/V entries needed by attention.
3. **Kernel launch and synchronization overhead.** Tiny kernels can be dominated by setup rather than arithmetic.

The rough scaling intuition is still useful:

- dense projections and MLPs scale mostly as `d²`;
- normalization and elementwise work scale mostly as `d`;
- cached attention and cache reads scale as `t·d`.

That is why elementwise work often occupies a larger **fraction** of latency in narrow models, while long-context cache traffic can reappear as a bottleneck even in wide models.

kipply cites a 336M-parameter, `d=1024` study where memory-bound intermediate operations made up roughly 43% of latency. Scaling width from 1024 to 8192 makes the `d²` work grow faster than the `d` work, but one should not simply divide 43% by eight and treat the result as a prediction. Fractions, kernel fusion, tensor shapes, and implementations all change together.

This is where fused LayerNorm/activation kernels and FlashAttention-style IO-aware attention help: they reduce HBM round trips rather than merely reducing arithmetic.

---

## Part 10 — Reality check against a 13B FasterTransformer run

kipply also reported a FasterTransformer benchmark for a 13B-shaped model:

| Quantity         |    Value |
| ---------------- | -------: |
| `d_model`        |    5,120 |
| Layers           |       40 |
| Heads            | 40 × 128 |
| Context          |      512 |
| Generated tokens |       10 |

The six-matrix baseline is

$$
24 \cdot 40 \cdot 5120^2
= 25.17\text{ GFLOPs/token}.
$$

### 10.1 Decode, one GPU

Ideal weight-bandwidth time:

$$\frac{25.17\text{e}9\text{ bytes}}{1.5\text{e}12\text{ bytes/s}} = 16.8\text{ ms}.$$

Reported measurement: **22.0ms per decode step**.

Using about 90% of peak HBM bandwidth raises the weight estimate to roughly 18.6ms. Profiled intermediate operations contributed about 2.2ms, and small remaining costs—launches, embeddings, and sampling—closed most of the gap.

### 10.2 Decode, two GPUs

The ideal weight floor halves to about **8.4ms**, but communication appears. The reported measurement was **13.5ms**, not half of 22ms. Smaller per-GPU tensors reached less bandwidth, intermediate kernels remained, and profiled communication contributed roughly 1.7ms.

This is the real tensor-parallel tradeoff in one result: weight time falls, but not every other term falls with it.

### 10.3 Prefill, 512 tokens

The ideal dense-matmul compute time is

$$
\frac{25.17\text{e}9 \cdot 512}{312\text{e}12}
\approx 41.3\text{ ms}.
$$

The reported one-GPU context time was about **63.2ms**. At this larger matrix-matrix shape, actual kernels did not reach peak tensor-core throughput: the profile observed roughly 72% of peak for an MLP matmul and roughly 54% for an attention projection. Prefill also performs causal attention and writes the initial KV cache.

### 10.4 The five recurring theory-to-reality gaps

1. sustained HBM bandwidth is below the spec-sheet peak;
2. intermediate and elementwise kernels are not free;
3. kernel launches, sampling, and synchronization add fixed costs;
4. real collectives have startup, topology, and bandwidth inefficiencies;
5. matmul efficiency depends on exact dimensions and tiling.

The arithmetic tells us **which regime to investigate**. Profiling tells us the constants for one model, runtime, and machine.

---

## Part 11 — A compact serving checklist

When sizing a deployment, calculate in this order.

### 1. Weight capacity

$$M_{\text{weights}} = P \cdot b_{\text{weight}}.$$

This gives the minimum accelerator count before runtime headroom.

### 2. KV-cache cost

$$
M_{\text{KV/token}}
= 2b_{\text{cache}}n_{\text{layers}}n_{\text{kv-heads}}d_{\text{head}}.
$$

Multiply by the sum of all live prompt and generated tokens—not merely the request count.

### 3. Dense roofline

$$
T_{\text{dense}}
\approx
\max\left(
\frac{M_{\text{weights}}}{N\cdot BW_{\text{HBM}}},
\frac{F_{\text{dense/token}}\cdot B}{N\cdot R_{\text{FLOP}}}
\right).
$$

This predicts whether batching can still amortize weight reads.

### 4. Context-dependent attention

Estimate both

$$F_{\text{attention/decode}} \approx 4t\,d\,n_{\text{layers}}$$

and the K/V bytes read. Long contexts can change a weight-bound decode into a cache-bandwidth-bound decode.

### 5. Tensor-parallel communication

Estimate payload size, collective count, topology, and startup latency. Then benchmark because collective efficiency is highly implementation-specific.

### 6. Validate with the real stack

Measure prefill latency, time to first token, inter-token latency, throughput, HBM use, and tail latency under the request-length distribution you actually expect.

---

## TL;DR — Key takeaways {#tldr}

- **Q is used once; K and V are reused by later tokens.** Causal masking makes cached K/V immutable after computation.
- **KV-cache memory is linear in live token count.** For ordinary BF16 MHA it is `4 · layers · d_model` bytes per token; GQA replaces attention-head count with the smaller KV-head count.
- **The cache avoids repeated prefix projections, not the attention scan.** Cached decode still reads past K/V and attends over a growing context.
- **Weights remain in HBM but are streamed through on-chip memory every step.** Reusing one weight stream across many token positions is why batching improves throughput.
- **`peak FLOPs ÷ HBM bandwidth` is the dense-matmul ridge point.** For the assumed A100 numbers it is about 208 FLOPs/byte, corresponding ideally to about 208 token positions per weight read.
- **Capacity and throughput are connected.** HBM left after weights determines how many live KV tokens can be retained, which constrains concurrency and batch formation.
- **Tensor parallelism divides weight traffic and math but introduces collectives.** Standard Megatron-style blocks have two logical activation reductions; each collective may contain multiple network phases.
- **`FLOPs ≈ 2P` is a baseline, not the whole step.** Cached attention, the vocabulary head, normalization, cache movement, and sampling remain.
- **More GPUs are not always faster.** Per-GPU work shrinks while activation communication does not shrink at the same rate.
- **Use arithmetic to classify the bottleneck; use benchmarks to schedule production.** Clean formulas are excellent for direction and poor substitutes for profiling exact model shapes.

---

## References

- [kipply, "Transformer Inference Arithmetic"](https://kipp.ly/p/transformer-inference-arithmetic)
- [kipply, "Transformer Parameter Counting"](https://kipp.ly/p/transformer-param-count)
- [NVIDIA A100 Tensor Core GPU architecture](https://www.nvidia.com/en-us/data-center/a100/)
- [NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) — tensor-parallel communication structure
- [Ivanov et al., "Data Movement Is All You Need"](https://arxiv.org/abs/2007.00072)
- [Dao et al., "FlashAttention"](https://arxiv.org/abs/2205.14135)
