---
layout: post
title: "Transformer Inference Arithmetic — A Worked Walkthrough"
subtitle: "KV-cache memory, batching, tensor parallelism, communication, and latency—derived with one 52B example."
date: 2026-09-16
categories: [Tech, machine-learning, llm]
tags: [machine-learning, llm, transformers, inference, gpu]
reading_time: 36
description: "A first-principles walkthrough of Transformer inference arithmetic: KV-cache size, weight bandwidth, batching, tensor parallelism, communication, FLOPs, and real-hardware corrections."
featured: true
---

_My notes on [kipply's "Transformer Inference Arithmetic"](https://kipp.ly/p/transformer-inference-arithmetic), rebuilt from first principles with one consistent example, full derivations, and no magic numbers._

**Want the short version? [Jump straight to the TL;DR.](#tldr)**

---

## Why this matters

Imagine an inference server receiving prompts from many users. For each request, it must first read the prompt, remember the useful attention state, and then generate new tokens one at a time. While doing that, it repeatedly moves tens of gigabytes of weights, grows a per-request cache, and—if the model spans several GPUs—exchanges partial results between devices.

That description gives us four resources to account for:

1. **weight memory** — can the model be loaded at all?
2. **KV-cache memory** — how many live tokens from active requests can be remembered?
3. **compute and HBM bandwidth** — how quickly can a forward step run?
4. **inter-GPU communication** — how much does splitting the model cost?

The arithmetic answers practical questions:

- How many requests can fit in one batch?
- What does one more token of context cost?
- Is this step limited by math, HBM bandwidth, or GPU-to-GPU communication?
- Will adding another GPU reduce latency, or only add communication?

The goal is not to predict latency to the microsecond. It is to build a model simple enough to calculate by hand and accurate enough to identify the right bottleneck.

### The story we will follow

Every section extends the same story rather than starting over:

1. text becomes token vectors;
2. a Transformer block turns those vectors into context-aware vectors;
3. the prompt is processed once (**prefill**) and its keys and values are cached;
4. new tokens are generated sequentially (**decode**);
5. requests are batched so they can share one pass over the weights;
6. the model is split across GPUs when one GPU cannot hold it;
7. a real benchmark shows which idealizations break first.

If a term in that list is unfamiliar, the glossary below defines it before any formulas appear.

---

## Part 0 — Vocabulary and notation

### 0.1 Model words

**Inference** means using an already-trained model to produce an output. Training changes the model's parameters; inference keeps them fixed and repeatedly applies them.

**Token** means one item from the model's vocabulary. A token may be a whole word, part of a word, punctuation, whitespace, or even a byte-like fragment. The tokenizer turns text into integer token IDs before the neural network runs.

**Vector** means an ordered list of numbers. Inside the model, one token is represented by a vector of width `d_model`. In our example, that is a list of 8,192 numbers.

**Embedding** means both the lookup table that maps token IDs to vectors and, informally, the resulting vectors. The table has one learned row per vocabulary item.

**Matrix** means a rectangular grid of numbers. Most model parameters live in large matrices. Multiplying a token vector by one of these matrices transforms the vector into a new representation.

**Parameter** (or **weight**) means one learned number stored in the model. "52B parameters" means roughly 52 billion learned numbers—not 52 billion operations and not 52GB.

**FLOP** means one floating-point operation. By the convention used in GPU specifications, one multiplication plus one addition counts as two FLOPs. A fused multiply-add instruction may execute both together, but it is still reported as two operations.

**Transformer block** means one repeated structural unit containing attention, an MLP, normalization, residual connections, and related operations. People often call a whole block a "layer," even though the block itself contains several linear layers.

**Self-attention** is the mechanism that lets one token read information from other token positions. It decides both _which_ earlier positions matter and _what_ information to gather from them.

**Residual stream** is the `d_model`-wide vector carried from block to block. Each attention or MLP sublayer adds an update to this stream instead of replacing it outright; that addition is a **residual connection**.

**MLP** (multilayer perceptron) is the per-token feed-forward network inside each block. Unlike attention, it does not mix information between token positions. In this example it expands each vector from 8,192 to 32,768 values and projects it back.

**LayerNorm** rescales a token's vector into a numerically well-behaved range. It has little arithmetic compared with a large matrix multiplication, but it still moves data and therefore takes time.

**LM head** is the final projection from the residual-stream width to one score per vocabulary item. Those raw scores are called **logits**. Softmax converts them into probabilities from which the next token can be selected.

### 0.2 Serving and hardware words

**Prefill** is the first forward pass over all tokens in a prompt. The entire prompt is known, so many token positions can be processed in parallel. Prefill creates the initial KV cache and largely determines **time to first token**.

**Decode** is the loop that follows prefill. Each iteration generates one new token per active request, appends its K/V state to the cache, and then starts the next iteration. Tokens within one response remain sequential: token 101 cannot be generated until token 100 is known.

**KV cache** stores the keys and values already computed for live token positions. It trades memory for avoiding repeated prefix computation.

**Batching** means processing token positions from multiple requests together. During decode, a batch of 100 usually means 100 active sequences each contributing one current token—not 100 future tokens from one sequence.

**Latency** is elapsed time for an individual request or step. **Throughput** is aggregate work completed per second across the server. Waiting briefly to form a larger batch may improve throughput while slightly worsening one request's latency.

**GPU** means the complete accelerator package. **HBM** (high-bandwidth memory, often called VRAM) is the GPU's attached large-capacity memory. **On-chip memory and compute units** are much smaller and faster structures inside the GPU where arithmetic actually happens.

**Capacity** answers "how many bytes fit?" **Bandwidth** answers "how many bytes move per second?" A 40GB GPU has a capacity of 40GB; its 1.5TB/s HBM bandwidth is a transfer rate, not another amount of storage.

**Tensor parallelism** splits individual weight matrices across GPUs. Every GPU performs part of a matrix multiplication. Some sharded intermediates can feed directly into the next local operation; a collective is required when partial sums must be combined or a replicated residual-stream result is needed.

### 0.3 Symbols used throughout

| Symbol            | Meaning                                                        |
| ----------------- | -------------------------------------------------------------- |
| `d` or `d_model`  | width of one token's residual-stream vector                    |
| `V`               | vocabulary size                                                |
| `L` or `n_layers` | number of Transformer blocks                                   |
| `S`               | number of token positions processed in a full sequence/prefill |
| `t`               | number of earlier positions visible at the current decode step |
| `B`               | token positions processed together in one batched operation    |
| `N`               | number of GPUs in the tensor-parallel group                    |
| `b`               | bytes used to store one number                                 |

The distinction between `S`, `t`, and `B` matters. Sequence length tells us how far attention looks; batch size tells us how many current token positions share the same weight read.

### 0.4 Our running example

| Quantity                                   |                   Value |
| ------------------------------------------ | ----------------------: |
| Model size                                 |          52B parameters |
| `d_model` (one token vector's width)       |                   8,192 |
| `n_layers` (stacked Transformer blocks)    |                      64 |
| `n_heads × d_head`                         |      `64 × 128 = 8,192` |
| `vocab_size` (distinct token IDs)          |                  50,257 |
| Weight/cache format                        | BF16, 2 bytes per value |
| GPU                                        |        NVIDIA A100 40GB |
| Peak BF16 compute                          |    312 trillion FLOPs/s |
| HBM bandwidth                              |    1.5 trillion bytes/s |
| Effective one-direction GPU link bandwidth |     300 billion bytes/s |

This is the same illustrative 52B shape used in kipply's post. It assumes ordinary multi-head attention and a two-matrix MLP with a `4d` hidden width. Modern models may use grouped-query attention (GQA), gated MLPs, RoPE, quantized weights, or different accelerators. Those change constants, not the method.

> **Important:** hardware peak numbers produce **lower bounds**, not promises. Unless stated otherwise, GB and TB below are decimal units, while "MiB" and "GiB" are binary units.

---

## Part 1 — What happens when the model produces a token

### 1.1 Text becomes IDs, then vectors

Suppose the prompt is:

> The capital of France is

The tokenizer might divide it into several pieces and assign each piece an integer ID. The exact split depends on the tokenizer. These IDs are addresses, not semantic vectors; ID 42 is not "twice as meaningful" as ID 21.

An embedding table

$$E \in \mathbb{R}^{V \times d}$$

maps each ID to one learned row. If token `i` has ID `id_i`, its initial vector is simply

$$x_i = E[\text{id}_i].$$

In our example, each row contains 8,192 BF16 values. For an `S`-token prompt, stacking those rows gives

$$X \in \mathbb{R}^{S \times d} = \mathbb{R}^{S \times 8192}.$$

The model also needs position information; otherwise the same words in different orders would look like an unordered set. Some architectures add learned position vectors. Many modern models instead apply RoPE while constructing queries and keys. Either way, the model gains a notion of order.

### 1.2 One pass through the complete model

From top to bottom:

1. **Tokenize:** text becomes `S` integer IDs.
2. **Embed:** each ID selects one `d`-wide vector.
3. **Run 64 Transformer blocks:** every block reads and writes an `S × d` residual stream.
4. **Apply the final LayerNorm:** the shape remains `S × d`.
5. **Apply the LM head:** the last useful hidden vector is projected from width `d` to width `V`.
6. **Select a token:** softmax turns the `V` logits into probabilities; greedy decoding, top-k, top-p, or another strategy chooses one ID.
7. **Append and repeat:** that ID becomes part of the sequence for the next decode step.

<img src="/assets/img/transformer-param-count/01-architecture.svg" alt="Decoder-only Transformer: tokenization and embedding, repeated attention and MLP blocks, then a vocabulary head." style="width: 100%; max-width: 54rem; height: auto; display: block; margin: 1.5rem auto;" />

The residual-stream shape is deliberately stable:

$$[S,d]\rightarrow[S,d]\rightarrow\cdots\rightarrow[S,d].$$

That `S × d` picture describes prefill. During cached decode, each block computes residual-stream updates only for the newest position from each active request, while attention reads old K/V from the cache. In both phases, every live token vector remains width `d`.

Only the LM head changes the width from `d` to `V`, because only there do we need one score for every possible next token. During decode, the server normally needs logits only for the newest position—not a fresh prediction from every old position.

### 1.3 Inside one Transformer block

Ignoring small architectural variations, a modern pre-normalized block is approximately:

$$
\begin{aligned}
u &= x + \operatorname{Attention}(\operatorname{LayerNorm}(x)),\\
y &= u + \operatorname{MLP}(\operatorname{LayerNorm}(u)).
\end{aligned}
$$

Read those equations as a sequence:

1. normalize the incoming residual stream;
2. let attention gather information from other positions;
3. add that update back to the original stream;
4. normalize again;
5. transform each token independently with the MLP;
6. add that update back too.

The residual additions are why each block can contribute an update without changing the `d_model` width. Exact normalization placement differs by architecture, but the large-matrix arithmetic below is unchanged by that detail.

Under this post's simplified architecture, the block's six large matrices are:

- attention: `Wq`, `Wk`, `Wv`, and `Wo`, each `d × d`;
- MLP: `W1`, shape `d × 4d`, and `W2`, shape `4d × d`.

Attention mixes information **between token positions**. The MLP transforms each position **independently**. LayerNorm, residual additions, positional operations, masking, activations, and softmax also run; they contain far fewer parameters but still move data, so they are not necessarily free in wall-clock time.

### 1.4 Self-attention, one step at a time

For one block, the residual stream is projected three different ways:

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,$$

where

$$W_Q,W_K,W_V\in\mathbb{R}^{d\times d}.$$

A useful—though imperfect—mental model is:

- **query:** what information is this position looking for?
- **key:** what kind of information does this position offer?
- **value:** what content should this position contribute if selected?

The most common misunderstanding is worth removing explicitly:

> `q · k` does **not** produce `v`. It produces one scalar relevance score. The value vector was computed independently using `Wv`.

For one current position and one attention head:

1. take the current query `q`;
2. dot it with every allowed key `k_i`, producing one score per visible position;
3. divide by `√d_head` to keep score magnitudes numerically stable;
4. mask future positions so they cannot be read;
5. softmax the scores into non-negative weights that sum to one;
6. multiply each `v_i` by its weight and add the results.

In compact notation,

$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}}+M_{\text{causal}}\right),\qquad Z=AV.$$

The model runs this mechanism in 64 heads. Each head projects from the full 8,192-wide residual stream into its own 128-wide Q/K/V output slice and can specialize in different relationships. The heads partition the **projected output width**, not the original input features. Their outputs are concatenated back to width 8,192 and mixed through `Wo`.

### 1.5 Prefill and decode use the same model differently

This distinction drives most inference arithmetic.

**During prefill**, all `S` prompt tokens are known. For one head:

$$Q,K,V\in\mathbb{R}^{S\times d_{\text{head}}},\qquad QK^\top\in\mathbb{R}^{S\times S}.$$

The causal mask hides the upper triangle, but the prompt's positions can still be processed in large parallel kernels. Prefill creates K and V for every prompt position.

**During decode**, only one new token per request is known at a time. If `t` earlier positions are already cached:

$$q_{\text{new}}\in\mathbb{R}^{1\times d_{\text{head}}},\qquad K_{\text{cache}},V_{\text{cache}}\in\mathbb{R}^{t\times d_{\text{head}}}.$$

The block first computes K and V for the current position. Attention temporarily combines them with the `t` prior entries, so the new query produces a `1 × (t+1)` score row and is allowed to attend to itself. The current K/V pair then becomes part of the persistent cache. The logits from that position predict the **following** token, which starts the next iteration.

Prefill is highly parallel within one request. Decode is sequential within one request but can be parallel **across many requests** through batching.

---

## Part 2 — The KV cache, quantified

### 2.1 Why K and V persist, but Q does not

For a token at position `i`, its query is needed while computing position `i`'s attention output. A later position `t` creates its own query, so it has no use for `q_i`.

The same later position _does_ need `k_i` and `v_i`: it compares `q_t` with `k_i`, then uses the resulting weight to decide how much of `v_i` to gather. Every future position may repeat that read.

Causality makes storage safe. At a given layer, position `i` cannot see positions after `i`; therefore adding a future token cannot change the K or V already computed for position `i`. For that request and layer, those vectors are immutable.

<img src="/assets/img/transformer-inference-arithmetic/01-kv-cache.svg" alt="A token's query is used once and discarded, while its key and value persist in the cache and are reused by future queries." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

The cache is therefore memoization: keep an expensive intermediate result because later steps will request exactly the same result. It does not approximate attention or change model quality.

### 2.2 Bytes stored per token

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

Each factor answers a different question:

- `2_K,V`: two vectors are retained;
- `2_bytes/value`: BF16 stores each component in two bytes;
- `64_layers`: every block has its own attention state;
- `64_heads × 128_values/head`: one complete 8,192-wide K or V vector per layer.

Because `n_heads · d_head = d_model`, the BF16 formula simplifies to

$$\boxed{\text{KV bytes/token} = 4\,n_{\text{layers}}d_{\text{model}}.}$$

That means one 2,048-token request needs

$$2\text{ MiB/token} \times 2048 = 4\text{ GiB}$$

of KV cache across the tensor-parallel group.

That 4GiB belongs to **one** 2,048-token sequence. Eight such live requests would need about 32GiB in this simplified model, even before they generate another token. A scheduler must therefore count the sum of cached tokens across all active requests:

$$M_{\text{KV,total}}=M_{\text{KV/token}}\sum_r t_r.$$

Under tensor parallelism, the cache is normally sharded by KV heads. Each GPU stores only its shard, while the values above describe the aggregate cache across the group.

For GQA or multi-query attention, replace `n_heads` with the smaller `n_kv_heads`:

$$\text{KV bytes/token} = 2b\,n_{\text{layers}}n_{\text{kv-heads}}d_{\text{head}}.$$

This is one reason serving-oriented models use fewer KV heads.

### 2.3 Projection FLOPs saved by caching

For one token at one layer, `Wk` and `Wv` are each `d × d`. Applying one such matrix costs approximately `2d²` FLOPs, so applying both across all layers costs

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

This does **not** mean a KV cache always reduces latency by exactly one-sixth. At position `t`, the important saving is that the server does not recompute old prefix state repeatedly. Meanwhile, the cache introduces memory reads of its own, and the current token still needs a complete pass through the model.

At decode position `t`, the cache avoids repeating those K/V projections for the previous `t` positions. It still:

1. computes K and V for the **new** token;
2. reads old K/V from the cache;
3. computes the new query against the `t` old keys plus its current key;
4. forms a weighted sum of the old values plus its current value.

### 2.4 A complexity correction worth remembering

It is tempting to say that a KV cache changes generation from quadratic to linear. That is not quite right.

- Without a cache, each decode step reruns projections over the whole prefix.
- With a cache, dense projection/MLP work is done only for the new token.
- But the new query still scans `t` cached keys and values.

So the **attention scan per decode step remains `O(t)`**. For a prompt of length `S` followed by `n` decode steps, the prior-context scans total

$$nS+\frac{n(n-1)}{2}=\Theta(nS+n^2).$$

The cache removes repeated prefix computation; it does not make attention itself constant-time.

Why is it still transformative? The expensive `d²` projections and MLP work no longer repeat for every old position. Only the context-dependent `t·d` attention scan grows each step. For wide models and ordinary context lengths, avoiding repeated `d²` work is an enormous reduction even though the remaining sequence-length complexity is not linear overall.

In practice, for a very wide model and moderate context, streaming the dense weights can dominate enough that consecutive decode steps look almost flat. At long contexts, KV-cache reads and attention become visible.

---

## Part 3 — Where compute actually happens

The phrase "load the model" hides two very different data movements.

**Event 1: storage → HBM.** At server startup, the checkpoint moves from disk or host memory into GPU HBM. This can take seconds, but it happens before requests are served, so it is not part of normal per-token latency.

**Event 2: HBM → on-chip memory and compute units.** During every forward pass, kernels fetch small tiles of weights into caches, shared memory, and registers close to the arithmetic units. The checkpoint remains resident in HBM, but the GPU cannot keep the entire model in its much smaller on-chip storage.

<img src="/assets/img/transformer-inference-arithmetic/04-hardware-data-path.svg" alt="Model weights move from storage to GPU HBM once at startup, then weight tiles stream from HBM through small on-chip memory to compute units on every forward pass." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

_[Open the hardware data-path diagram at full size](/assets/img/transformer-inference-arithmetic/04-hardware-data-path.svg)._

Think of HBM as a warehouse next to a factory. Loading the checkpoint stocks the warehouse once. Serving a token still requires bringing each needed pallet to the factory floor. The same weights remain in the warehouse between requests, but most cannot remain beside the arithmetic units.

Real GPUs reuse tiles while a kernel works and may retain a small fraction in cache. "Streaming the weights" is a first-order model of the aggregate traffic: the model is far larger than on-chip storage, so a decode pass must fetch roughly one model's worth of dense weights from HBM.

For 52B BF16 parameters:

$$52\text{e}9\text{ parameters}\times2\text{ bytes/parameter}=104\text{e}9\text{ bytes}=104\text{ GB}.$$

On a hypothetical single A100 with enough capacity, the ideal bandwidth floor for reading those bytes once would be

$$T_{\text{weights,1}} = \frac{104\text{e}9}{1.5\text{e}12} = 69.3\text{ ms}.$$

The real model does not fit on one 40GB A100. With four-way tensor parallelism, each GPU reads roughly 26GB:

$$T_{\text{weights,4}} = \frac{26\text{e}9}{1.5\text{e}12} = 17.3\text{ ms}.$$

These are ideal **weight-read floors**, not full step latencies. Attention, cache reads, communication, and software overhead are added later.

### 3.1 Why batching amortizes weight traffic

Start with one BF16 weight. Reading it costs two bytes. Using it for one token contributes roughly one multiplication and one addition: two FLOPs. That is only about one FLOP per byte, far below the A100's available compute-to-bandwidth balance.

Now let `B` token positions use the same matrix together. The matrix is still fetched once for the operation, but every weight is reused across `B` rows of activations. Weight traffic stays roughly fixed while useful arithmetic scales with `B`:

$$
\text{arithmetic intensity} \approx
\frac{2BP\text{ FLOPs}}{2P\text{ bytes}}
= B\text{ FLOPs/byte}.
$$

Arithmetic intensity means "useful work performed per byte fetched." The A100 can ideally sustain the following hardware balance:

$$\frac{312\text{e}12\text{ FLOP/s}}{1.5\text{e}12\text{ byte/s}} \approx 208\text{ FLOPs/byte}.$$

Equating the workload's `B` FLOPs/byte with the hardware's 208 FLOPs/byte gives the ideal dense-matmul **ridge point** near `B=208`:

- **Below 208:** weights cannot arrive quickly enough to keep all arithmetic units busy. The operation is memory-bandwidth-bound.
- **Near 208:** weight delivery and arithmetic take similar time. Hardware utilization is best in this simplified model.
- **Above 208:** the arithmetic units are full, so adding more positions increases step time. The operation is compute-bound.

For decode, `B` usually means concurrent sequences contributing one new token each. It does **not** mean the server generates 208 sequential future tokens from one request simultaneously. Prefill also processes many positions together, although attention and exact matrix shapes make it less identical to simple batching than this roofline suggests.

<img src="/assets/img/transformer-inference-arithmetic/02-batching-crossover.svg" alt="Ideal batching roofline for the 52B model on four A100 GPUs: a 17.3 millisecond weight-streaming floor meets the compute line near a batch of 208." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

_[Open the batching crossover graph at full size](/assets/img/transformer-inference-arithmetic/02-batching-crossover.svg)._

Using only the six large matrices:

| Positions processed together (`B`) | Weight floor, TP=4 | Ideal math time, TP=4 | Dense-matmul lower bound |
| ---------------------------------: | -----------------: | --------------------: | -----------------------: |
|                                  1 |            17.3 ms |              0.083 ms |                  17.3 ms |
|                                 50 |            17.3 ms |               4.13 ms |                  17.3 ms |
|                                100 |            17.3 ms |               8.26 ms |                  17.3 ms |
|                            **208** |        **17.3 ms** |           **17.2 ms** |             **≈17.3 ms** |
|                                500 |            17.3 ms |               41.3 ms |                  41.3 ms |

The tiny 17.3ms-versus-17.2ms mismatch comes from using the rounded full 52B weight footprint for the memory line but the 51.54B block-matrix count for the math line. Using the same set of dense weights in numerator and denominator makes the ideal crossover exactly the hardware ratio, 208; it does not change the regime analysis.

Read the last column as

$$T_{\text{dense}}\approx\max(T_{\text{weight read}},T_{\text{math}}),$$

because optimized matrix kernels overlap fetching tiles with computing on earlier tiles. This does **not** make additional positions literally free below 208. It means the idealized dense matmuls use arithmetic capacity that would otherwise wait on weight delivery. KV reads, attention, activation traffic, communication, imperfect kernels, and batching overhead give the supposedly flat line a slope in reality.

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

Three GPUs are only the **capacity floor**. They are not automatically a usable or efficient tensor-parallel configuration. Our 64 attention heads divide cleanly across 4 or 8 GPUs but not 3, and many GPU servers have power-of-two link topologies. Four GPUs are therefore the more natural configuration for this example.

With four GPUs, the 56GB remainder is aggregate memory—about 14GB per GPU before runtime overhead. Because attention heads and their cache are sharded, each device stores its corresponding fraction.

As a concrete workload, eight requests with 2,048 cached tokens each consume

$$8\times2048\times2\text{ MiB}=32\text{ GiB}.$$

They fit below the theoretical four-GPU ceiling, but generation grows every request's cache one token at a time. The scheduler must leave room for that growth or evict, swap, or reject work.

These are **ceilings**, not safe scheduler limits. A serving runtime also needs:

- temporary activations and workspaces;
- communication buffers;
- allocator headroom and memory lost to fragmentation;
- CUDA context and kernel-library allocations;
- possibly untied vocabulary-head weights.

Consequently, a practical four-GPU limit can be noticeably below 26,700 cached tokens. This is not a rounding issue; it is reserved and fragmented memory.

Paged KV-cache allocators reduce fragmentation by managing cache memory in blocks rather than demanding one large contiguous region per request. They improve utilization but cannot exceed the physical byte budget.

The scheduling consequence is still clear: after the weights fit, extra HBM can increase concurrency, and concurrency lets the server build larger efficient batches. Capacity is therefore not separate from throughput; it determines how many requests are available to batch.

---

## Part 5 — Splitting weights across GPUs

Tensor parallelism shards each large matrix. With four GPUs, each package owns roughly one quarter of the weight bytes and performs roughly one quarter of the matmul work:

$$104\text{ GB}/4=26\text{ GB of weights per GPU}.$$

Because every GPU has its own HBM channels and compute units, this is more than pooling capacity: four devices can stream four shards and multiply them in parallel.

<img src="/assets/img/transformer-inference-arithmetic/03-tensor-parallel.svg" alt="Four tensor-parallel GPUs each read a 26GB weight shard, compute a partial output, and exchange partials in a collective to reconstruct the full activation." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

### 5.1 Why communication is required

Imagine splitting a long arithmetic sum among four people. Each person can calculate one quarter independently, but nobody has the final total until the four partial sums are combined. Sharded matrix multiplication has the same dependency.

Depending on which matrix dimension is partitioned, each GPU may produce:

- a distinct slice that can be concatenated; or
- a partial sum that must be reduced across GPUs.

A common Megatron-style Transformer block arranges the sharding so that it needs **two logical activation all-reduces per block**:

1. after attention's output projection;
2. after the MLP's down-projection.

An all-reduce is itself implemented as multiple network phases—for example, reduce-scatter plus all-gather. This is why some descriptions count four communication phases per block. Calling all four phases "four all-reduces" would overcount the logical synchronization points.

Why can the other projections avoid immediate synchronization? `Wq`, `Wk`, `Wv`, and the MLP up-projection can produce sharded intermediate features that the same GPU continues processing locally. The output projection `Wo` and MLP down-projection `W2` convert those shards back into residual-stream updates; their partial sums must be combined before the next dependent operation.

### 5.2 A simple communication-volume model

One residual-stream activation contains `B × d` BF16 values, so its payload is

$$M_{\text{payload}} = 2Bd\text{ bytes}.$$

For `B=500` and `d=8192`,

$$M_{\text{payload}} = 2 \cdot 500 \cdot 8192 = 8.192\text{ MB}.$$

For a ring all-reduce over `N` GPUs, a common estimate for bytes **sent per GPU** is

$$M_{\text{ring}} \approx 2\frac{N-1}{N}M_{\text{payload}}.$$

The factor `2(N-1)/N` accounts for data sent during the reduce-scatter and all-gather phases. Each GPU receives the same amount concurrently on a full-duplex link. It approaches two sent payloads per GPU as the group grows.

At `N=4`, that is about **12.288 MB per logical all-reduce**. Two reductions per layer across 64 layers move about 1.57GB per GPU:

$$12.288\text{ MB} \times 2 \times 64 \approx 1.57\text{ GB}.$$

At an idealized effective 300GB/s, the bandwidth term is

$$T_{\text{comm,volume}} \approx \frac{1.57\text{e}9}{300\text{e}9} = 5.24\text{ ms}.$$

This is only the bandwidth term. Every collective also has startup latency, and each layer must wait for required results before proceeding. The exact number depends on topology, collective algorithm, whether bandwidth is quoted per direction or aggregate, message size, and overlap.

---

## Part 6 — A latency model that does not contradict itself

There is no single "model latency." A serving system usually tracks at least:

- **prefill latency / time to first token:** how long until generation begins;
- **inter-token latency:** time between consecutive output tokens for one request;
- **batch step time:** how long one scheduler iteration takes;
- **throughput:** total output tokens completed per second across all requests.

A useful step-time decomposition is

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

Inside one optimized matrix kernel, weight reads and arithmetic overlap, which motivates the `max` in the dense roofline. Between dependent Transformer stages, however, a required collective can remain on the critical path. Real runtimes overlap whatever they safely can, so the final answer lies between perfect overlap and complete serialization.

### 6.1 Small decode batch

For `B=1`, TP=4:

- ideal weight-streaming floor: **17.3ms**;
- ideal math time: **0.083ms**;
- collective startup: if one logical collective costs an optimistic `8μs`, then

$$2 \cdot 64 \cdot 8\mu\text{s} \approx 1.0\text{ ms}.$$

If that collective startup is fully exposed, the illustrative serialized estimate is **17.3 + 1.0 ≈ 18.3ms** before long-context cache reads and runtime overhead. With perfect overlap, the corresponding bound is about 17.3ms, so this toy model gives a **17.3–18.3ms range**. The `8μs` figure is only an assumption; real collective latency depends strongly on generation, topology, and implementation.

### 6.2 Large batch: `B=500`

The dense math term is

$$
\frac{103.08\text{e}9 \cdot 500}{4 \cdot 312\text{e}12}
\approx 41.3\text{ ms}.
$$

The ring-volume estimate from Part 5 is about **5.24ms**. There are also `2 × 64 = 128` collective startups. Writing startup latency as `α`, a serialized estimate is

$$41.3\text{ ms}+5.24\text{ ms}+128\alpha.$$

With the same toy `α=8μs` assumption, this is about **47.6ms**, plus attention, cache, and software overhead. With useful overlap it could be closer to 41.3ms; with inefficient collectives it could be higher than 47.6ms.

For scale, 500 separate idealized `B=1` steps at 18.3ms would occupy about

$$500\times18.3\text{ ms}=9.15\text{ s},$$

whereas one `B=500` step is on the order of 47.6ms before omitted costs—roughly a 192× gain in aggregate work per unit time.

That comparison means **500 active requests each produce one next token**. It does not mean one request produces 500 sequential tokens in 46.5ms. Batching improves aggregate throughput; it cannot remove the dependency between consecutive tokens of the same response.

---

## Part 7 — When does communication become the bottleneck?

Picture each GPU as a factory and the interconnect as a fleet of trucks. A factory computes one partial activation; the trucks exchange those partials so every factory can continue with the combined result. If local computation takes much longer than communication, transfer is a small tax. If sharding makes each factory's job tiny while the shipment stays activation-sized, the trucks become the limiting resource.

The A100's compute-to-link ratio is

$$\frac{312\text{e}12\text{ FLOP/s}}{300\text{e}9\text{ byte/s}} \approx 1040\text{ FLOPs/byte}.$$

This is a second roofline. A stage that performs fewer than about 1040 useful FLOPs per communicated byte is at risk of becoming communication-bound.

Focus first on attention's row-parallel output projection, `Wo`. Each GPU multiplies an input slice of width `d/N` into a full-width partial output. For `B` positions, that is

$$F_{\text{Wo,GPU}}=\frac{2Bd^2}{N}\text{ FLOPs}.$$

Part 5's ring all-reduce sends

$$M_{\text{ring}}=2\frac{N-1}{N}(2Bd)=4Bd\frac{N-1}{N}\text{ bytes per GPU}.$$

Dividing the local work by the bytes it must send gives

$$I_{\text{Wo,link}}=\frac{F_{\text{Wo,GPU}}}{M_{\text{ring}}}=\frac{d}{2(N-1)}\text{ FLOPs/byte}.$$

The MLP down-projection starts from width `4d`, so it performs four times as much local math for the same `B × d` residual-stream payload:

$$I_{\text{W2,link}}=\frac{2d}{N-1}\text{ FLOPs/byte}.$$

Applying those formulas:

| Tensor-parallel size | `Wo` intensity | `W2` intensity | Comparison with 1040 FLOPs/byte                |
| -------------------: | -------------: | -------------: | ---------------------------------------------- |
|                    4 |          1,365 |          5,461 | `Wo` has a narrow theoretical compute cushion  |
|                    8 |            585 |          2,341 | `Wo` communication is exposed                  |
|                   16 |            273 |          1,092 | `Wo` is strongly exposed; `W2` is near balance |

This is not a universal cutoff. Faster links, topology-aware collectives, overlap, quantized communication, and different matrix shapes move it. But it explains the tradeoff:

> More GPUs reduce each GPU's weight traffic and math, while making the fixed-size activation exchange large relative to each GPU's shrinking share of work.

The minimum GPU count is a capacity question. The best GPU count is a latency-throughput-cost question.

Notice the direction of both trends:

- increasing `N` divides weight traffic and dense FLOPs by more GPUs;
- the residual-stream activation being combined does not shrink at the same rate;
- startup latency also does not disappear.

That is why adding GPUs gives diminishing returns even before cost is considered.

---

## Part 8 — Where "FLOPs per token ≈ 2 × parameters" comes from

### 8.1 The one matmul rule

Multiplying an `m × n` matrix by a length-`n` vector performs approximately

$$2mn\text{ FLOPs},$$

counting one multiply and one add per matrix element. The matrix itself contains `mn` parameters. Therefore, if a weight is used once in a matmul,

$$\text{FLOPs} \approx 2 \times \text{parameters}.$$

A tiny example makes the rule concrete. Let

$$
W=
\begin{bmatrix}
w_{11}&w_{12}&w_{13}\\
w_{21}&w_{22}&w_{23}
\end{bmatrix},
\qquad
x=
\begin{bmatrix}
x_1\\x_2\\x_3
\end{bmatrix}.
$$

`W` contains `2 × 3 = 6` parameters. Producing `Wx` uses every parameter once: six multiplications and approximately six additions, or about 12 FLOPs. The exact elementary count is `2mn-m` because the first term in each sum needs no preceding addition; at dimensions like 8,192, the difference from `2mn` is negligible. Hardware specifications also conventionally report a multiply-accumulate as two FLOPs.

### 8.2 The six large matrices in one block

| Matrix | Shape          |  Parameters | FLOPs/token |
| ------ | -------------- | ----------: | ----------: |
| `Wq`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wk`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wv`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wo`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `W1`   | `8192 × 32768` | 268,435,456 | 536,870,912 |
| `W2`   | `32768 × 8192` | 268,435,456 | 536,870,912 |

The 64 attention heads do not create another factor of 64 here. `Wq`, for example, can be viewed as 64 narrower `8192 × 128` projections placed side by side:

$$64\times8192\times128=8192\times8192=d^2.$$

Heads partition the output width; their widths sum back to `d_model`.

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

**Cached attention.** With `t` earlier positions plus the current position, QK scores and the weighted-V operation add roughly

$$4(t+1)d\,n_{\text{layers}}\approx4td\,n_{\text{layers}}\text{ FLOPs}.$$

At `t=2048` earlier positions, that is

$$
4 \cdot 2049 \cdot 8192 \cdot 64
\approx 4.30\text{ GFLOPs},
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

For example, LayerNorm must at least read a token vector, compute statistics, and write a normalized vector. Even if each element needs only a handful of operations, the bytes still travel. If LayerNorm, a residual addition, and an activation are separate kernels, an intermediate vector may make several avoidable HBM round trips.

Kernel fusion keeps an intermediate tile on-chip while applying several operations. It does not change the mathematical model or its parameter count; it reduces traffic between kernels.

KV-cache traffic follows a different scaling rule. At context length 2,048, our one request owns 4GiB of cached K/V across all layers. A decode step attends over those earlier positions layer by layer, so cache reads can become a material bandwidth term even though no weights are added.

The rough scaling intuition is still useful:

- dense projections and MLPs scale mostly as `d²`;
- normalization and elementwise work scale mostly as `d`;
- cached attention and cache reads scale as `t·d`.

That is why elementwise work often occupies a larger **fraction** of latency in narrow models, while long-context cache traffic can reappear as a bottleneck even in wide models.

kipply cites a 336M-parameter, `d=1024` study where memory-bound intermediate operations made up roughly 43% of latency. Scaling width from 1024 to 8192 makes the `d²` work grow faster than the `d` work, but one should not simply divide 43% by eight and call the answer 5%.

Even in an unrealistically simple two-component model, the original ratio is

$$r=\frac{T_{\text{linear}}}{T_{\text{quadratic}}}=\frac{0.43}{0.57}\approx0.75.$$

Increasing `d` eightfold would reduce that ratio to about `0.75/8≈0.094`, corresponding to a new share of `0.094/(1+0.094)≈8.6%`. That calculation is still only directional: fractions, kernel fusion, attention length, tensor shapes, and implementations all change together.

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

The comparison in one view:

| Workload                 |       Simple lower bound | Reported measurement | What became visible                                           |
| ------------------------ | -----------------------: | -------------------: | ------------------------------------------------------------- |
| 1-GPU decode             |                   16.8ms |               22.0ms | sustained HBM bandwidth, intermediate kernels, fixed overhead |
| 2-GPU decode             | 8.4ms plus communication |               13.5ms | smaller-shard efficiency and collectives                      |
| 1-GPU, 512-token prefill |        41.3ms dense math |               63.2ms | sub-peak matmul efficiency, attention, and cache writes       |

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

$$F_{\text{attention/decode}} \approx 4(t+1)d\,n_{\text{layers}}\approx4td\,n_{\text{layers}}$$

and the K/V bytes read. Long contexts can change a weight-bound decode into a cache-bandwidth-bound decode.

### 5. Tensor-parallel communication

Estimate payload size, collective count, topology, and startup latency. Then benchmark because collective efficiency is highly implementation-specific.

### 6. Validate with the real stack

Measure prefill latency, time to first token, inter-token latency, throughput, HBM use, and tail latency under the request-length distribution you actually expect.

---

## TL;DR — Key takeaways {#tldr}

- **A request has two phases.** Prefill processes the known prompt in parallel and creates the initial cache; decode generates one new token per active request per iteration.
- **Q is used once; K and V are reused by later tokens.** Causal masking makes cached K/V immutable after computation.
- **KV-cache memory is linear in live token count.** For ordinary BF16 MHA it is `4 · layers · d_model` bytes per token; GQA replaces attention-head count with the smaller KV-head count.
- **The cache avoids repeated prefix projections, not the attention scan.** Cached decode still reads past K/V and attends over a growing context.
- **Weights remain in HBM but are streamed through on-chip memory every step.** Reusing one weight stream across many token positions is why batching improves throughput.
- **`peak FLOPs ÷ HBM bandwidth` is the dense-matmul ridge point.** For the assumed A100 numbers it is about 208 FLOPs/byte, corresponding ideally to about 208 token positions per weight read.
- **During decode, batch size means concurrent current positions.** It does not let one response generate sequential future tokens in parallel.
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
