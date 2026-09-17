---
layout: post
title: "Transformer Inference Arithmetic — A Worked Walkthrough"
subtitle: "How KV cache, batching, memory bandwidth, and multi-GPU serving fit together—derived step by step from one 52B model and one running workload."
date: 2026-09-16
categories: [Tech, machine-learning, llm]
tags: [machine-learning, llm, transformers, inference, gpu]
reading_time: 48
description: "A beginner-first but complete walkthrough of Transformer inference: what problem each mechanism solves, every derivation worked out on a single 52B example, and what the numbers mean when you serve it on four A100s."
featured: true
---

_My notes on [kipply's "Transformer Inference Arithmetic"](https://kipp.ly/p/transformer-inference-arithmetic), rebuilt around one question: what makes an LLM request use memory, compute, and time?_

**Want the short version? [Jump straight to the TL;DR.](#tldr)**

---

## The question this post answers

Imagine eight people chatting with the same model. Each has already sent a 2,048-token prompt, and the server is generating the next token for all eight conversations.

What must the server do?

1. keep the model's learned numbers in GPU memory;
2. remember useful attention data for every token in every conversation;
3. move the model's weights to the compute units on each decode step;
4. combine partial results if the model is split across GPUs.

Those four jobs create the four costs we will explain: **weight memory, KV-cache memory, computation, and communication**.

The important part is how they connect:

> Model size decides whether the weights fit. The memory left after the weights fit decides how many conversations can stay active. Active conversations create the batch. The batch decides whether weight movement or arithmetic is slower. Splitting across GPUs helps the first two problems but adds communication.

Every formula below answers one question in that chain, and every number comes from the same model and the same eight conversations, so each part builds on the previous one instead of restarting.

### How to read this post

Every numbered part opens with a **Question**, and Parts 1–10 close with a **Carry forward** note that hands the unsolved problem to the next part. Those two lines are the story: read only them and you still get a connected argument from prompt to benchmark.

Inside each part, subsections marked **Going deeper** contain the full derivation—the algebra, the exact FLOP counts, the second-order costs. They are there on purpose: the conclusions should never feel like magic, and you should be able to push as far into the technical detail as you want. Skip them on a first pass; come back when you want to know _why_ a number is what it is.

---

## Part 0 — The minimum vocabulary

These are the only terms needed to start:

- A **token** is one text piece understood by the model. It can be a word, part of a word, punctuation, or whitespace.
- A **vector** is a list of numbers. One token is represented inside this model by 8,192 numbers.
- An **embedding** is the learned lookup that turns a token ID into that first vector.
- A **parameter** or **weight** is one learned number in the model. Most weights are arranged in rectangular grids called **matrices**.
- A **matmul** (matrix multiplication) multiplies a grid of weights by the token vectors. It is where nearly all the arithmetic happens.
- A **FLOP** is one floating-point operation. One multiplication plus one addition counts as roughly two FLOPs.
- **Dense** weights are the ordinary full weight matrices that _every_ token passes through. In this post "dense" and "the six large matrices" mean the same thing.
- A **Transformer block** (or layer) is one repeated processing unit. **Attention** lets token positions gather information from one another. The **MLP** (multilayer perceptron) widens each token vector, transforms it, and shrinks it back.
- The **residual stream** is the running `tokens × 8,192` representation that every block reads and writes.
- A **kernel** is one GPU program, such as a single matmul or normalization.
- **Prefill** processes the known prompt. **Decode** generates new tokens afterward, one step at a time.
- A **KV cache** remembers attention data from earlier token positions so decode does not rebuild it.
- A **batch** contains token positions processed together. During decode, batch size `B=8` usually means eight conversations each producing one current token.
- **Latency** is how long one request or step waits. **Throughput** is how many total tokens the server produces per second.
- **HBM** (high-bandwidth memory) is the GPU's large attached memory. **Capacity** asks how many bytes fit; **bandwidth** asks how many bytes can move per second.
- **Tensor parallelism** splits one model across several GPUs. Each GPU computes a piece, and some pieces must then be combined.
- **BF16** (bfloat16) is the two-byte number format used for weights and cache values in this example.

The recurring symbols are:

| Symbol            | Meaning                                       | Value here |
| ----------------- | --------------------------------------------- | ---------: |
| `d` or `d_model`  | width of one token vector                     |      8,192 |
| `L` or `n_layers` | number of Transformer blocks                  |         64 |
| `n_heads`         | attention heads per block                     |         64 |
| `d_head`          | width of one head                             |        128 |
| `V`               | vocabulary size                               |     50,257 |
| `b`               | bytes per stored number                       |          2 |
| `S`               | prompt length in tokens                       |      2,048 |
| `t`               | earlier tokens visible during one decode step |      2,048 |
| `B`               | token positions processed together            |          8 |
| `N`               | GPUs sharing the model                        |          4 |
| `P`               | parameter count                               |        52B |

### Our model and workload

The main walkthrough uses one illustrative 52B shape. Part 10 switches to a published 13B benchmark only to compare theory with a real measurement.

| Model or hardware quantity       |                                          Value |
| -------------------------------- | ---------------------------------------------: |
| Parameters                       |                                     52 billion |
| Token-vector width, `d_model`    |                                          8,192 |
| Transformer blocks, `n_layers`   |                                             64 |
| Attention                        | 64 query heads and 64 KV heads, each width 128 |
| MLP hidden width                 |                                  `4d = 32,768` |
| Vocabulary size                  |                                         50,257 |
| Weight and cache format          |                        BF16, 2 bytes per value |
| GPU                              |                               NVIDIA A100 40GB |
| Peak BF16 compute                |                           312 trillion FLOPs/s |
| HBM bandwidth                    |                           1.5 trillion bytes/s |
| One-direction GPU-link bandwidth |                            300 billion bytes/s |

Our running workload is:

- **4 A100 GPUs** sharing the model;
- **8 active conversations**;
- **2,048 cached tokens per conversation**;
- therefore **`B=8`** during a decode step.

This is an intentionally simple teaching example, not a claim that it is the best production configuration. Peak compute and bandwidth rates are upper limits on what the hardware can do, so every time estimate derived from them is an optimistic lower bound.

GB uses decimal powers of 1,000; GiB and MiB use binary powers of 1,024. The distinction only changes small numerical details here, but the labels are kept explicit.

---

## Part 1 — From a prompt to one next token

> **Question:** What work is the server repeating when it generates text?

Suppose a user writes:

> The capital of France is

The model cannot read text directly. The request follows this path:

1. **Tokenize:** split the text into token pieces and replace each piece with an integer ID.
2. **Embed:** use each ID to look up an 8,192-number vector.
3. **Add position information:** tell the model where each token appears, because word order matters.
4. **Run 64 Transformer blocks:** each block refines the token vectors.
5. **Produce vocabulary scores:** turn the final vector into 50,257 scores, one for every possible next token.
6. **Choose one token:** append it to the sequence and repeat the process.

<img src="/assets/img/transformer-param-count/01-architecture.svg" alt="Decoder-only Transformer: tokenization and embedding, repeated attention and MLP blocks, then a vocabulary head." style="width: 100%; max-width: 54rem; height: auto; display: block; margin: 1.5rem auto;" />

For a prompt containing `S` tokens, the model carries an `S × 8,192` grid of numbers through the blocks. Each block keeps the same outer shape. It adds two kinds of updates:

- **attention** gathers useful information from other token positions;
- the **MLP** transforms each position on its own.

**Normalization** rescales a vector's values into a stable range. A **residual addition** adds a block's update to the incoming vector instead of replacing it. Both help information flow through 64 blocks without changing the 8,192-wide shape.

### 1.1 Going deeper: text becomes IDs, then vectors

The tokenizer divides the text into pieces and assigns each piece an integer ID. These IDs are addresses, not meanings; ID 42 is not "twice as meaningful" as ID 21.

An embedding table

$$E \in \mathbb{R}^{V \times d}$$

maps each ID to one learned row. If token `i` has ID `id_i`, its initial vector is simply

$$x_i = E[\text{id}_i].$$

Each row holds 8,192 BF16 values. Stacking the rows for an `S`-token prompt gives

$$X \in \mathbb{R}^{S \times d} = \mathbb{R}^{S \times 8192}.$$

The model also needs position information; otherwise the same words in any order would look like an unordered set. Some architectures add learned position vectors. Many modern models instead apply **RoPE** (rotary position embeddings) while constructing queries and keys. Either way, the model gains a notion of order.

The residual-stream shape is deliberately stable through the whole stack:

$$[S,d]\rightarrow[S,d]\rightarrow\cdots\rightarrow[S,d].$$

Only the final **LM head** changes the width from `d` to `V`, because only there do we need one score per possible next token. During decode the server normally needs those scores for the newest position only—not a fresh prediction from every old position.

### 1.2 Going deeper: inside one block

Ignoring small architectural variations, a modern pre-normalized block is

$$
\begin{aligned}
u &= x + \operatorname{Attention}(\operatorname{LayerNorm}(x)),\\
y &= u + \operatorname{MLP}(\operatorname{LayerNorm}(u)).
\end{aligned}
$$

Read those two lines as a sequence:

1. normalize the incoming residual stream;
2. let attention gather information from other positions;
3. add that update back to the original stream;
4. normalize again;
5. transform each position independently with the MLP;
6. add that update back too.

The residual additions are why each block can contribute an update without changing the 8,192-wide shape. Exact normalization placement differs by architecture, but none of the arithmetic below depends on that detail.

Under this post's architecture, the block's six large matrices are:

- attention: `Wq`, `Wk`, `Wv`, and `Wo`, each `d × d`;
- MLP: `W1`, shape `d × 4d`, and `W2`, shape `4d × d`.

Attention mixes information **between token positions**. The MLP transforms each position **independently**. Normalization, residual additions, positional operations, masking, activation functions, and softmax also run; they hold far fewer parameters but still move data, so they are not free in wall-clock time. Part 9 returns to exactly that point.

### 1.3 Why attention creates Q, K, and V

Each block makes three different projections of the token vectors:

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,\qquad W_Q,W_K,W_V\in\mathbb{R}^{d\times d}.$$

A **projection** simply means multiplying by one of those matrices to create a new set of vectors.

A useful—though imperfect—mental model is:

- **query:** what is the current position looking for?
- **key:** what does an earlier position advertise about itself?
- **value:** what content can that earlier position contribute?

The most common misunderstanding is worth removing explicitly:

> `q · k` does **not** produce `v`. It produces one scalar relevance score. The value vector was computed independently, using `Wv`.

For one current position and one attention head:

1. take the current query `q`;
2. dot it with every allowed key `k_i`, producing one score per visible position;
3. divide by `√d_head` so score magnitudes do not grow with head width and saturate the softmax into a near one-hot choice;
4. mask future positions so they cannot be read;
5. softmax the scores into non-negative weights that sum to one;
6. multiply each `v_i` by its weight and add the results.

In compact notation,

$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}}+M_{\text{causal}}\right),\qquad Z=AV.$$

The model runs this mechanism in 64 **heads** so different heads can specialize in different relationships. Each head projects from the full 8,192-wide residual stream into its own 128-wide Q/K/V slice. The heads partition the **projected output width**, not the original input features; their outputs are concatenated back to width 8,192 and mixed through `Wo`. That detail matters in Part 8, where it explains why 64 heads do _not_ multiply the FLOP count by 64.

### 1.4 Prefill and decode use the same model differently

This distinction drives most inference arithmetic.

**During prefill**, all `S` prompt tokens are known. For one head:

$$Q,K,V\in\mathbb{R}^{S\times d_{\text{head}}},\qquad QK^\top\in\mathbb{R}^{S\times S}.$$

The causal mask hides the upper triangle, but every prompt position can still be processed in large parallel kernels. Prefill also creates K and V for every prompt position.

**During decode**, only one new token per request is known at a time. If `t` earlier positions are already cached:

$$q_{\text{new}}\in\mathbb{R}^{1\times d_{\text{head}}},\qquad K_{\text{cache}},V_{\text{cache}}\in\mathbb{R}^{t\times d_{\text{head}}}.$$

The block first computes K and V for the current position. Attention combines them with the `t` prior entries, so the new query produces a `1 × (t+1)` score row—the current token is allowed to attend to itself. The current K/V pair then joins the persistent cache, and the resulting scores predict the **following** token, which starts the next iteration.

One conversation cannot generate all its future tokens in parallel: token 101 depends on token 100. Our eight conversations _can_ advance together, however. Each contributes one current position, giving the server a decode batch of `B=8`.

> **Carry forward:** Prefill is parallel within one request; decode is sequential within a request but parallel across requests. And earlier K and V vectors are needed again on every later decode step, so the server needs somewhere to remember them.

---

## Part 2 — The KV cache, quantified

> **Question:** Why does serving memory grow as conversations get longer?

For a token at position `i`, its query is needed while computing position `i`'s attention output. A later position `t` creates its own query, so it has no use for `q_i`.

The same later position _does_ need `k_i` and `v_i`: it compares `q_t` with `k_i`, then uses the resulting weight to decide how much of `v_i` to gather. Every future position may repeat that read.

Causality makes storage safe. At a given layer, position `i` cannot see positions after `i`; therefore adding a future token cannot change the K or V already computed for position `i`. For that request and layer, those vectors are immutable.

<img src="/assets/img/transformer-inference-arithmetic/01-kv-cache.svg" alt="A token's query is used once and discarded, while its key and value persist in the cache and are reused by future queries." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

The KV cache is therefore **memoization**: keep an intermediate result because later steps will request exactly the same result. It does not approximate attention and does not change the model's answer.

### 2.1 What one cached token costs

For the ordinary multi-head attention assumed in our model table, each token stores one K vector and one V vector at every layer:

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

- `2_K,V`: two vectors are retained per token per layer;
- `2_bytes/value`: BF16 stores each component in two bytes;
- `64_layers`: every block has its own attention state;
- `64_heads × 128_values/head`: one complete 8,192-wide K or V vector per layer.

Because `n_heads · d_head = d_model`, the BF16 formula collapses to a shortcut worth memorizing:

$$\boxed{\text{KV bytes/token} = 4\,n_{\text{layers}}d_{\text{model}}.}$$

One 2,048-token conversation therefore needs

$$2\text{ MiB/token} \times 2048 = 4\text{ GiB},$$

and our eight active conversations need

$$8\times4\text{ GiB}=32\text{ GiB}.$$

Every decode step adds one cached token per conversation, so the batch grows by another

$$8\times2\text{ MiB}=16\text{ MiB per step}.$$

This is why a memory manager counts **live tokens**, not requests:

$$M_{\text{KV,total}}=M_{\text{KV/token}}\sum_r t_r.$$

Eight short conversations and eight very long conversations have completely different cache costs. Under tensor parallelism the cache is normally sharded by KV heads, so each GPU stores its own fraction; the 32GiB above is the aggregate across the group.

### 2.2 Going deeper: exactly how much compute the cache saves

For one token at one layer, `Wk` and `Wv` are each `d × d`. Applying one such matrix costs about `2d²` FLOPs, so applying both across all layers costs

$$
2_{\text{matmul}}
\cdot 2_{\text{K,V}}
\cdot n_{\text{layers}}
\cdot d^2
= 4 \cdot 64 \cdot 8192^2
= 17{,}179{,}869{,}184
$$

or about **17.18 GFLOPs per old token**. The six large matrices in the full block stack cost about 103.08 GFLOPs per token (derived in Part 8), so K and V projections are exactly **one-sixth** of that dense baseline.

Now price that on our own workload. Without a cache, every decode step would re-project K and V for all 2,048 earlier positions in all eight conversations:

$$17.18\text{ GFLOPs}\times2048\times8\approx281.5\text{ TFLOPs},$$

which even on four ideal A100s would take

$$\frac{281.5\text{e}12}{4\times312\text{e}12}\approx225.5\text{ ms}.$$

And K/V projections are only one-sixth of the story. A truly cacheless implementation reruns the _whole_ dense stack for every position it still needs:

$$\frac{103.08\text{ GFLOPs}\times 8\times2049}{4\times312\text{e}12}\approx1{,}354\text{ ms}.$$

Compare that with the 0.66ms of ideal arithmetic a cached `B=8` step needs (Part 3). **The cache turns roughly 1.35 seconds of repeated math per step into under a millisecond**—about a 2,000× reduction—at the price of the 32GiB we just counted. That single trade is the reason the rest of this post is mostly about memory.

What the cache does _not_ remove: the current token still computes its own K and V, still reads the old K/V from memory, still scores its query against `t+1` keys, and still forms a weighted sum over `t+1` values.

### 2.3 Going deeper: the complexity correction worth remembering

It is tempting to say a KV cache turns generation from quadratic into linear. That is not quite right.

- Without a cache, each decode step reruns projections over the whole prefix.
- With a cache, dense projection and MLP work happens only for the new token.
- But the new query still scans `t` cached keys and values.

So the **attention scan per decode step remains `O(t)`**. For a prompt of length `S` followed by `n` decode steps, the prior-context scans total

$$nS+\frac{n(n-1)}{2}=\Theta(nS+n^2).$$

For one of our conversations generating 500 tokens after its 2,048-token prompt, that is

$$500\times2048+\frac{500\times499}{2}=1{,}148{,}750$$

key/value positions scanned per layer, per conversation—growing quadratically in output length even _with_ a perfect cache.

Why is the cache still transformative? Because the expensive `d²` projection and MLP work no longer repeats for old positions; only the `t·d` attention scan grows. For a wide model at ordinary context lengths, removing repeated `d²` work is an enormous win even though sequence-length complexity is not linear overall. At very long contexts, the surviving `t·d` term and its memory traffic become visible again—exactly what we will measure in Part 6.

### 2.4 Going deeper: GQA, the knob that shrinks the cache

Our model stores 64 KV heads, one per query head. Grouped-query and multi-query attention let several query heads share one K/V pair, so the formula becomes

$$\text{KV bytes/token} = 2b\,n_{\text{layers}}n_{\text{kv-heads}}d_{\text{head}}.$$

If our 52B model kept 64 query heads but only 8 KV heads, each token would cost

$$2\cdot2\cdot64\cdot8\cdot128=262{,}144\text{ bytes}=256\text{ KiB},$$

one-eighth of 2MiB. Our eight conversations would then hold **4GiB instead of 32GiB** of cache. Keep that number in mind: it reappears in Part 4 as the difference between serving 13 conversations and serving 104, and in Part 6 as the difference between 5.7ms and 0.7ms of cache reads per step. This is why serving-oriented models overwhelmingly use fewer KV heads.

> **Carry forward:** The cache solves a compute problem by creating a memory problem. We already need 32GiB for eight conversations, and we have not yet counted the model's weights.

---

## Part 3 — Why decode often waits on memory, not math

> **Question:** An A100 can perform 312 trillion operations per second. Why can generating one token still take many milliseconds?

The phrase "load the model" hides two very different data movements.

**Event 1: storage → HBM.** At server startup, the checkpoint moves from disk or host memory into GPU HBM. This can take seconds, but it happens before requests are served, so it is not part of normal per-token latency.

**Event 2: HBM → on-chip memory and compute units.** During every forward pass, kernels fetch small tiles of weights into caches, shared memory, and registers close to the arithmetic units. The checkpoint stays resident in HBM, but the GPU cannot keep a 104GB model in its much smaller on-chip storage.

<img src="/assets/img/transformer-inference-arithmetic/04-hardware-data-path.svg" alt="Model weights move from storage to GPU HBM once at startup, then weight tiles stream from HBM through small on-chip memory to compute units on every forward pass." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

_[Open the hardware data-path diagram at full size](/assets/img/transformer-inference-arithmetic/04-hardware-data-path.svg)._

Think of HBM as a warehouse beside a factory. Startup stocks the warehouse once. Every decode step still brings the needed pallets to the factory floor. The weights remain on the GPU, but they are not all beside the arithmetic units.

Real GPUs reuse each tile while a kernel works and keep a small fraction in cache, so "streaming the weights" is a first-order model of aggregate traffic: because the model is far larger than on-chip storage, a decode pass must fetch roughly one model's worth of dense weights from HBM.

For 52B BF16 parameters:

$$52\text{e}9\text{ parameters}\times2\text{ bytes/parameter}=104\text{e}9\text{ bytes}=104\text{ GB}.$$

On a hypothetical single A100 with enough capacity, reading those bytes once would take at least

$$T_{\text{weights,1}} = \frac{104\text{e}9}{1.5\text{e}12} = 69.3\text{ ms}.$$

The model does not actually fit on one 40GB A100. Across four GPUs, each GPU owns and reads roughly 26GB:

$$T_{\text{weights,4}} = \frac{26\text{e}9}{1.5\text{e}12} = 17.3\text{ ms}.$$

Now compare that with arithmetic. The six large matrices need about 103.08 billion FLOPs per token; Part 8 derives that number. Split across four ideal A100s:

$$T_{\text{math,token}}=\frac{103.08\text{e}9}{4\times312\text{e}12}\approx0.083\text{ ms},$$

so for our eight-conversation batch

$$T_{\text{math},B=8}=8\times0.083\approx0.66\text{ ms}.$$

The GPUs need 0.66ms of ideal arithmetic but about 17.3ms to stream their weight shards—a factor of 26. **Weight movement is the bottleneck**, and the arithmetic units sit idle waiting for data. This is the problem batching solves.

### 3.1 Why batching amortizes weight traffic

Start with one BF16 weight. Reading it costs two bytes. Using it for one token contributes roughly one multiplication and one addition: two FLOPs. That is about **1 FLOP per byte**, far below what the A100 can sustain.

Now let `B` token positions use the same matrix together. The matrix is fetched once for the operation, but every weight is reused across `B` rows of activations. Weight traffic stays roughly fixed while useful arithmetic scales with `B`. Writing `P` for the number of weights in the matrix:

$$
\text{arithmetic intensity} \approx
\frac{2BP\text{ FLOPs}}{2P\text{ bytes}}
= B\text{ FLOPs/byte}.
$$

**Arithmetic intensity** means useful operations performed per byte moved. In plain language: a batch of 8 gets eight uses out of each fetched weight; a batch of 208 gets 208 uses.

The A100's own compute-to-bandwidth balance is

$$\frac{312\text{e}12\text{ FLOP/s}}{1.5\text{e}12\text{ byte/s}} \approx 208\text{ FLOPs/byte}.$$

Equating the workload's `B` FLOPs/byte with the hardware's 208 FLOPs/byte puts the ideal **crossover** (also called the roofline ridge point) near `B=208`:

- **Below 208:** weights cannot arrive quickly enough to keep the arithmetic units busy. The operation is memory-bandwidth-bound.
- **Near 208:** weight delivery and arithmetic take similar time. Hardware utilization is best in this simplified model.
- **Above 208:** the arithmetic units are saturated, so adding positions increases step time. The operation is compute-bound.

During decode, `B` means concurrent conversations contributing one current token each. It does **not** mean one conversation generates 208 future tokens simultaneously.

<img src="/assets/img/transformer-inference-arithmetic/02-batching-crossover.svg" alt="Ideal batching roofline for the 52B model on four A100 GPUs: a 17.3 millisecond weight-streaming floor meets the compute line near a batch of 208." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

_[Open the batching crossover graph at full size](/assets/img/transformer-inference-arithmetic/02-batching-crossover.svg)._

The graph uses `TP=4` as shorthand for four-way tensor parallelism. Using only the six large matrices:

| Positions processed together (`B`) | Weight floor, TP=4 | Ideal math time, TP=4 | Large-matrix lower bound |
| ---------------------------------: | -----------------: | --------------------: | -----------------------: |
|                                  1 |            17.3 ms |              0.083 ms |                  17.3 ms |
|                       **8 (ours)** |        **17.3 ms** |           **0.66 ms** |              **17.3 ms** |
|                                 50 |            17.3 ms |               4.13 ms |                  17.3 ms |
|                                100 |            17.3 ms |               8.26 ms |                  17.3 ms |
|                            **208** |        **17.3 ms** |           **17.2 ms** |             **≈17.3 ms** |
|                                500 |            17.3 ms |               41.3 ms |                  41.3 ms |

Read the last column as

$$T_{\text{large matrices}}\approx\max(T_{\text{weight read}},T_{\text{math}}),$$

because optimized matmul kernels compute on one tile while fetching the next; the slower stream sets the floor. That does **not** make extra positions below 208 literally free—they consume arithmetic capacity that would otherwise idle while weights arrive. Cache reads, attention, communication, imperfect kernels, and batching overhead give the supposedly flat line a real slope.

Two footnotes on the numbers. The tiny 17.3-versus-17.2 mismatch at `B=208` comes from using the rounded 52B weight footprint for the memory line but the 51.54B block-matrix count (Part 8.2) for the math line; using one consistent weight set makes the crossover land exactly on the hardware ratio. And the `B=500` row is purely counterfactual: four A100s could not hold 500 of our 2,048-token conversations, as Part 4 shows.

### 3.2 Going deeper: prefill is the opposite regime

The same model, the same hardware, and the same formulas classify prefill completely differently—because prefill has `B = S = 2048` positions available at once.

Dense math to prefill one 2,048-token prompt on four A100s:

$$\frac{103.08\text{e}9\times2048}{4\times312\text{e}12}\approx169\text{ ms},$$

against the same 17.3ms weight-read floor. Prefill is roughly **ten times compute-bound**, while our decode step was 26 times memory-bound. Its attention term is real but secondary: the `S × S` score matrices across all layers cost

$$4S^2dL=4\cdot2048^2\cdot8192\cdot64\approx8.8\text{ TFLOPs}$$

(about half that once the causal mask is exploited), which is roughly 4% of the dense prefill work, or 3.5–7ms.

That is why a serving system reports two different latencies: **time to first token** is dominated by compute-bound prefill (~169ms+ here), while **inter-token latency** is dominated by memory-bound decode (~23ms here, derived in Part 6).

> **Carry forward:** Our `B=8` decode is far below the ideal 208-position crossover, so more concurrent conversations would buy almost free throughput—but every additional conversation needs KV-cache memory.

---

## Part 4 — Capacity: does it fit?

> **Question:** We want a larger batch, but can the weights and all live caches fit in HBM together?

The weights alone need at least

$$\left\lceil\frac{104}{40}\right\rceil = 3\text{ A100-40GB GPUs}.$$

But "the weights fit" is not enough. Whatever remains must hold the KV cache and the runtime's temporary buffers. Dividing the remainder by 2,097,152 bytes per token gives:

| GPUs | Total HBM | Weight memory | Theoretical remainder | Theoretical KV-token capacity |
| ---: | --------: | ------------: | --------------------: | ----------------------------: |
|    3 |    120 GB |        104 GB |                 16 GB |                 ≈7,629 tokens |
|    4 |    160 GB |        104 GB |                 56 GB |                ≈26,703 tokens |

Our eight conversations already contain

$$8\times2{,}048=16{,}384\text{ live tokens}.$$

So three GPUs can hold the weights but **cannot hold our workload's cache**. Four GPUs can: the cache uses 32GiB (about 34.4GB in decimal units), leaving roughly 21.6GB before runtime overhead.

Three GPUs are only the _capacity floor_, and they are awkward for another reason: our 64 attention heads divide cleanly across 4, 8, or 16 GPUs but not 3, and most GPU servers have power-of-two link topologies. Four GPUs are the natural configuration here.

### 4.1 Capacity is a throughput question in disguise

The 26,703-token ceiling limits the batch we can form. If every conversation is 2,048 tokens long, four GPUs can theoretically retain only

$$\left\lfloor\frac{26{,}703}{2{,}048}\right\rfloor=13\text{ such conversations}.$$

A decode batch near 13 is nowhere near the 208-position crossover from Part 3. **Long contexts consume exactly the memory that would otherwise buy concurrency, and concurrency is what makes weight streaming efficient.** Capacity is therefore not a separate topic from throughput; it decides how many requests even exist to batch.

And 13 is a mathematical ceiling, not a safe scheduler limit. A real runtime also needs:

- temporary activations and kernel workspaces;
- communication buffers;
- allocator headroom and memory lost to fragmentation;
- CUDA context and kernel-library allocations;
- possibly untied vocabulary-head weights.

**Paged KV-cache allocators** reduce fragmentation by managing the cache in fixed blocks rather than one large contiguous region per request. They improve utilization but cannot exceed the physical byte budget.

Now recall the GQA variant from Part 2.4. Holding the weights fixed at 104GB, the same 56GB remainder at 256KiB per token instead of 2MiB holds

$$
\frac{56\text{e}9}{262{,}144}\approx213{,}600\text{ tokens}
\quad\Longrightarrow\quad
\frac{213{,}600}{2048}\approx104\text{ conversations}.
$$

The same hardware, one architectural decision—and the reachable decode batch moves from 13 to about 104, into the neighborhood of the 208-position crossover. That is the whole argument for grouped-query attention in one line of arithmetic.

That 104 is deliberately conservative, because fewer KV heads also shrink the weights themselves. With 8 KV heads, `Wk` and `Wv` become `8192 × 1024` instead of `8192 × 8192`, removing

$$2\cdot64\cdot8192\cdot(8192-1024)\approx7.52\text{B parameters}=15.0\text{ GB},$$

which would leave about 71GB for cache and push the ceiling past 130 conversations. Real designs usually spend those freed parameters elsewhere—more layers or a wider MLP—so holding the model at 52B is the fairer comparison and the safer estimate.

> **Carry forward:** Four GPUs solve our capacity problem. But no single GPU now computes the full answer, so the devices must exchange partial results.

---

## Part 5 — Splitting weights across GPUs

> **Question:** If four GPUs each compute only part of a matrix multiplication, how does the model recover one correct answer?

Tensor parallelism shards each large matrix. With four GPUs, each package owns roughly one quarter of the weight bytes and performs roughly one quarter of the matmul work:

$$104\text{ GB}/4=26\text{ GB of weights per GPU}.$$

Because every GPU has its own HBM channels and compute units, this is more than pooling capacity: four devices can stream four shards and multiply them in parallel. That is precisely how Part 3's 69.3ms single-device weight floor became 17.3ms.

<img src="/assets/img/transformer-inference-arithmetic/03-tensor-parallel.svg" alt="Four tensor-parallel GPUs each read a 26GB weight shard, compute a partial output, and exchange partials in a collective to reconstruct the full activation." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

### 5.1 Why communication is required, and exactly how often

Imagine splitting a long arithmetic sum among four people. Each can compute one quarter independently, but nobody has the final total until the four partial sums are combined. Sharded matrix multiplication has the same dependency.

Depending on which matrix dimension is partitioned, each GPU produces either:

- a distinct slice of the output that can simply be concatenated; or
- a **partial sum** that must be added across GPUs.

The operation that exchanges and sums partial answers is called an **all-reduce**. A Megatron-style block is arranged so that only **two logical all-reduces per block** are needed:

1. after attention's output projection `Wo`;
2. after the MLP's down-projection `W2`.

Why do the other four matrices escape synchronization? `Wq`, `Wk`, `Wv`, and the MLP up-projection `W1` are column-parallel: each GPU produces its own slice of heads or hidden units and keeps processing that slice locally. `Wo` and `W2` are row-parallel—they consume a sharded input and produce a full-width residual-stream update, so their partial sums must be combined before the next dependent operation reads the residual stream.

With 64 blocks, one decode step performs

$$2\times64=128\text{ logical all-reduces}.$$

Each all-reduce is itself implemented as multiple network phases—typically reduce-scatter followed by all-gather—which is why some descriptions count four communication phases per block. Calling those phases "four all-reduces" would overcount the synchronization points.

That sounds expensive, but communication has two separate costs, and which one dominates depends entirely on batch size.

### 5.2 Small messages pay startup cost

Starting a collective has a fixed latency even when its payload is tiny—like establishing a phone call before speaking.

For our `B=8` workload, one **activation payload**—the current `B × d` slice of the residual stream—is only

$$M_{\text{payload}}=2Bd=2\text{ bytes}\times8\times8192=131{,}072\text{ bytes}=128\text{ KiB}.$$

If one collective has an optimistic `8μs` startup cost, our 128 collectives contribute roughly

$$128\times8\mu\text{s}\approx1.0\text{ ms}.$$

At this batch size, fixed startup matters far more than the data itself. Two caveats on that 1.0ms. The `8μs` figure is an assumption, not a specification; real collective latency depends strongly on hardware generation, topology, and implementation. And it charges startup once per _logical_ all-reduce—if each of the two network phases pays its own message latency, the term doubles to roughly `4·64·8μs ≈ 2.0ms`, which is how kipply's own estimate is written.

### 5.3 Large messages also pay for bytes moved

For the hypothetical `B=500` point on the graph—reachable only with much shorter contexts or far more memory—one payload grows to

$$2\times500\times8192=8.192\text{ MB}.$$

For a ring all-reduce over `N` GPUs, a common estimate for bytes **sent per GPU** is

$$M_{\text{ring}} \approx 2\frac{N-1}{N}M_{\text{payload}}.$$

The factor `2(N-1)/N` accounts for data sent during the reduce-scatter and all-gather phases; each GPU receives the same amount concurrently on a full-duplex link, and the factor approaches 2 as the group grows. At `N=4` that is about **12.288MB per logical all-reduce**, so 128 collectives move

$$12.288\text{ MB}\times128\approx1.57\text{ GB per GPU},$$

and at an idealized 300GB/s the bandwidth term is

$$T_{\text{comm,volume}} \approx \frac{1.57\text{e}9}{300\text{e}9} \approx 5.24\text{ ms}.$$

So the shape of the cost is:

- **small batch:** mostly "start 128 exchanges" (≈1.0ms of startup, negligible bytes);
- **large batch:** startup **plus** moving much larger activations (≈5.24ms of volume at `B=500`).

Exact values depend on links, topology, collective algorithm, whether bandwidth is quoted per direction or aggregate, message size, and how much the runtime can overlap.

### 5.4 Why more GPUs do not keep halving latency

Moving from four to eight GPUs halves each weight shard from 26GB to 13GB, so the ideal weight-read floor falls from 17.3ms to about 8.7ms. But the 8,192-wide activation still has to be combined, now among more participants, and startup latency does not shrink at all.

More GPUs reduce local memory traffic and arithmetic. They do not remove collective startup or make the exchanged activation smaller. Eventually communication becomes large relative to each GPU's shrinking share of local work—Part 7 derives exactly where that happens.

> **Carry forward:** A decode step is not "compute time plus every other number." Weight reads and arithmetic overlap, while some communication is exposed between dependent stages. We need one latency model that keeps those relationships straight.

---

## Part 6 — Putting step latency together

> **Question:** Which costs overlap, which costs add delay, and what do they mean for one user versus the whole server?

There is no single "model latency." A serving system usually tracks at least four numbers:

- **prefill latency / time to first token:** how long until generation begins;
- **inter-token latency:** time between consecutive output tokens for one request;
- **batch step time:** how long one scheduler iteration takes;
- **throughput:** total output tokens completed per second across all requests.

A useful step-time decomposition is

$$
T_{\text{step}}
\approx T_{\text{large matrices}}
+ T_{\text{attention and cache}}
+ T_{\text{communication on the critical path}}
+ T_{\text{fixed overhead}},
$$

where the large-matrix term is the roofline from Part 3:

$$
T_{\text{large matrices}}
\approx
\max\left(
\frac{\text{weight bytes}}{N\cdot BW_{\text{HBM}}},
\frac{F_{\text{dense/token}}\cdot B}{N\cdot R_{\text{FLOP}}}
\right).
$$

We take the maximum inside a matmul because an optimized kernel computes on one tile while fetching another. But between dependent stages, a required collective can sit squarely on the critical path. The honest general statement is a sandwich:

$$
\max(T_{\text{compute}},T_{\text{comm}})
\leq T_{\text{compute+comm}}
\leq T_{\text{compute}}+T_{\text{comm}}.
$$

Neither "always add every term" nor "always take one maximum" is universally correct. Real runtimes overlap whatever they safely can, so the truth lies between perfect overlap and full serialization.

### 6.1 Our eight-conversation step

We already have every ingredient:

- weight-read floor, TP=4: **17.3ms**;
- ideal math for `B=8`: **0.66ms**;
- collective startup for 128 all-reduces: **≈1.0ms**, volume negligible at 128KiB payloads.

So the large-matrix term is about 17.3ms. Now include the long context, which is where our 32GiB of cache finally shows up as _time_ rather than bytes. During one decode step, standard attention reads the K/V state for all eight conversations. Sharded evenly across four GPUs, that is 8GiB per GPU, and at an ideal 1.5TB/s:

$$T_{\text{KV read}}\approx\frac{8\text{ GiB}}{1.5\text{ TB/s}}\approx5.7\text{ ms}.$$

Weights and cache contend for the same HBM bandwidth, so together they set a memory-traffic floor of

$$17.3+5.7=23.0\text{ ms}.$$

If the ~1.0ms of collective cost is fully exposed, the estimate becomes about **24.1ms** before software overhead (about 25.1ms if each collective's two phases each pay startup); with good overlap it stays near 23.0ms. Implementations change the exact traffic and overlap, but the calculation proves that long-context cache reads are not a rounding error—they are a quarter of our step.

The step produces eight next tokens, so the throughput ceiling is

$$\frac{8}{0.0230}\approx348\text{ tokens/s aggregate},$$

falling to about 332 aggregate tokens/s with fully exposed communication—roughly **41 tokens/s per continuously active conversation**. Real throughput will be lower once software overhead is included.

Note what the GQA variant would do to this same step: cache reads drop from 5.7ms to 0.7ms, so the memory floor falls from 23.0ms to about 18.0ms even before the smaller `Wk`/`Wv` shrink the weight term—and the _reachable_ batch grows eightfold. Both effects push the same direction.

### 6.2 Why a larger batch helps throughput but can hurt latency

Suppose the requests had much shorter contexts—or the server had more memory—so it could form `B=208`. The weight and math terms would both be about 17.3ms, so the large-matrix-only ceiling would approach

$$\frac{208}{0.0173}\approx12{,}000\text{ positions/s}$$

before cache, communication, and software costs. Compare the extremes directly. One idealized `B=1` step costs about 18.3ms—17.3ms of weight streaming plus ~1.0ms of exposed startup, with cache reads left out on both sides of this comparison—so serving 500 requests one at a time would occupy

$$500\times18.3\text{ ms}=9.15\text{ s},$$

whereas one `B=500` step costs on the order of 47.6ms (41.3ms of math, 5.24ms of ring volume, ~1.0ms of startup)—about a **192× gain in aggregate work per unit time**.

The catch is that this is aggregate work, not per-user speed. A request may wait while the scheduler forms the batch, and the step itself gets longer past the crossover. The tradeoff is:

- **larger batch:** better total hardware efficiency;
- **individual request:** potentially more queueing and a longer step.

Batching improves throughput only when enough live work exists _and_ the cache for that work fits—and `B=500` still means 500 requests each producing one token.

> **Carry forward:** Two loose threads remain. First, when does communication overtake compute as `N` grows? Second, where did 103.08 GFLOPs come from?

---

## Part 7 — When does communication become the bottleneck?

> **Question:** We saw that eight GPUs do not halve latency. Can we predict the point where adding GPUs stops helping?

Picture each GPU as a factory and the interconnect as a fleet of trucks. A factory computes one partial activation; the trucks exchange those partials so every factory can continue with the combined result. If local computation takes much longer than communication, transfer is a small tax. If sharding makes each factory's job tiny while the shipment stays activation-sized, the trucks become the limiting resource.

Part 3 compared compute against HBM bandwidth. Now compare compute against _link_ bandwidth:

$$\frac{312\text{e}12\text{ FLOP/s}}{300\text{e}9\text{ byte/s}} \approx 1040\text{ FLOPs/byte}.$$

This is a second roofline. A stage performing fewer than about 1040 useful FLOPs per communicated byte risks becoming communication-bound.

Apply it to attention's row-parallel output projection `Wo`. Each GPU multiplies an input slice of width `d/N` into a full-width partial output, so for `B` positions it performs

$$F_{\text{Wo,GPU}}=\frac{2Bd^2}{N}\text{ FLOPs},$$

while the ring all-reduce from Part 5.3 sends

$$M_{\text{ring}}=2\frac{N-1}{N}(2Bd)=4Bd\frac{N-1}{N}\text{ bytes per GPU}.$$

Dividing local work by the bytes it must send gives

$$I_{\text{Wo,link}}=\frac{F_{\text{Wo,GPU}}}{M_{\text{ring}}}=\frac{d}{2(N-1)}\text{ FLOPs/byte}.$$

Notice that `B` cancels: this is a **structural** property of the sharding, not something a bigger batch can fix. The MLP down-projection starts from width `4d`, so it performs four times as much local math for the same `B × d` payload:

$$I_{\text{W2,link}}=\frac{2d}{N-1}\text{ FLOPs/byte}.$$

For our `d=8192`:

| Tensor-parallel size | `Wo` intensity | `W2` intensity | Comparison with 1040 FLOPs/byte                |
| -------------------: | -------------: | -------------: | ---------------------------------------------- |
|                **4** |      **1,365** |      **5,461** | `Wo` has a narrow theoretical compute cushion  |
|                    8 |            585 |          2,341 | `Wo` communication is exposed                  |
|                   16 |            273 |          1,092 | `Wo` is strongly exposed; `W2` is near balance |

So our four-GPU configuration sits just above the line, and eight GPUs would push attention's output projection below it—which is exactly the diminishing return Part 5.4 predicted, now with a number attached.

This is not a universal cutoff. Faster links, topology-aware collectives, overlap with independent compute, quantized communication, and different matrix shapes all move it. But the direction is reliable:

> More GPUs reduce each GPU's weight traffic and math, while the fixed-size activation exchange does not shrink at the same rate—and startup latency does not shrink at all.

The minimum GPU count is a capacity question (Part 4). The best GPU count is a latency-throughput-cost question, and this roofline is how you bound it before benchmarking.

> **Carry forward:** Both rooflines—HBM in Part 3 and links here—compared arithmetic against bytes, and both used one unexplained number: 103.08 GFLOPs per token. Time to earn it.

---

## Part 8 — Where "FLOPs per token ≈ 2 × parameters" comes from

> **Question:** Why did we use 103.08 billion FLOPs per token in Parts 3, 6, and 2.2?

### 8.1 One small matrix explains the rule

Multiplying an `m × n` matrix by a length-`n` vector performs approximately

$$2mn\text{ FLOPs},$$

counting one multiply and one add per matrix element. The matrix itself contains `mn` parameters. Therefore, if a weight is used once in a matmul,

$$\text{FLOPs} \approx 2 \times \text{parameters}.$$

A tiny example makes it concrete. Let

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

`W` contains `2 × 3 = 6` parameters, and producing `Wx` uses every one of them exactly once: six multiplications and about six additions, so roughly 12 FLOPs. The exact elementary count is `2mn-m`, because the first term of each sum needs no preceding addition; at dimensions like 8,192 that correction is negligible, and hardware specifications conventionally report a multiply-accumulate as two FLOPs anyway.

### 8.2 The six large matrices in one block

| Matrix | Shape          |  Parameters | FLOPs/token |
| ------ | -------------- | ----------: | ----------: |
| `Wq`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wk`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wv`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wo`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `W1`   | `8192 × 32768` | 268,435,456 | 536,870,912 |
| `W2`   | `32768 × 8192` | 268,435,456 | 536,870,912 |

Grouped, that is:

| Part of one block | Large matrices                             | Parameters | FLOPs per token |
| ----------------- | ------------------------------------------ | ---------: | --------------: |
| Attention         | Q, K, V, and output: four `d × d` matrices |      `4d²` |           `8d²` |
| MLP               | one `d × 4d` and one `4d × d` matrix       |      `8d²` |          `16d²` |
| **Total**         | six matrices                               | **`12d²`** |      **`24d²`** |

The 64 attention heads do **not** add another factor of 64. `Wq`, for example, can be seen as 64 narrower `8192 × 128` projections placed side by side:

$$64\times8192\times128=8192\times8192=d^2.$$

Heads partition the output width; their widths sum back to `d_model`—the same fact that made the KV-cache formula collapse to `4·L·d` in Part 2.1.

One block therefore costs

$$
4(2d^2) + 2(2d\cdot4d)
= 24d^2
= 1{,}610{,}612{,}736\text{ FLOPs/token},
$$

and across 64 blocks:

$$
24 \cdot 64 \cdot 8192^2
= 103{,}079{,}215{,}104
\approx 103.08\text{ GFLOPs/token}.
$$

Dividing by two recovers the weight count of those matrices:

$$\frac{103.08\text{B}}{2} = 51.54\text{B parameters},$$

which is almost the entire 52B model. Hence the rule we have been leaning on:

$$\boxed{\text{dense FLOPs per token}\approx2\times\text{dense parameters}.}$$

### 8.3 What the `2P` shortcut leaves out

The 103.08 GFLOPs figure is the dense projection-plus-MLP baseline, not the whole decode step.

**Cached attention.** With `t` earlier positions plus the current one, the QK scores and the weighted-V sum add roughly

$$4(t+1)d\,n_{\text{layers}}\approx4td\,n_{\text{layers}}\text{ FLOPs}.$$

At our `t=2048`:

$$
4 \cdot 2049 \cdot 8192 \cdot 64
\approx 4.30\text{ GFLOPs},
$$

about **4.2%** of the six-matrix baseline, growing linearly with context on every decode step. Note the asymmetry we measured in Part 6: those same cached positions were only 4% of the _arithmetic_ but 25% of the _time_, because attention re-reads 8GiB per GPU while performing very little math per byte.

**Vocabulary head.** Projecting the final hidden state to 50,257 logits costs

$$
2Vd
= 2 \cdot 50{,}257 \cdot 8192
\approx 0.823\text{ GFLOPs}.
$$

The embedding table holds about 411.7M parameters. Adding them to the block matrices closes the loop on our model size:

$$51.54\text{B}+0.41\text{B}\approx51.95\text{B}\approx52\text{B}.$$

If embeddings are **tied**, those same parameters serve as the vocabulary head and _do_ participate in a matmul at output time, even though the input embedding is only a lookup. If untied, the model stores a second matrix of the same size.

**Other kernels.** Normalization, RoPE, residual additions, activation functions, softmax, sampling, and cache reads add smaller FLOP counts but nonzero latency—the subject of the next part.

So `2P` is powerful because most large-model parameters are used exactly once per token in a dense matmul. It is a baseline, not an exact law.

> **Carry forward:** The FLOP count is now fully derived. But Part 6.1 already showed that 4% of the arithmetic ate 25% of the step, so FLOPs alone clearly do not predict time.

---

## Part 9 — Memory traffic the FLOP count misses

> **Question:** If the arithmetic is ~4% attention and ~95% the six large matrices, why do profiles show time going elsewhere?

FLOP accounting makes normalization, softmax, residual additions, and activations look trivial. On a GPU, an operation can perform almost no arithmetic and still spend real time reading and writing tensors.

Three effects matter:

1. **Intermediate activation traffic.** Unfused kernels may write a tensor to HBM only for the next kernel to read it straight back.
2. **KV-cache traffic.** Each decode step reads all earlier K/V entries attention needs—our 8GiB per GPU.
3. **Kernel launch and synchronization overhead.** Tiny kernels can be dominated by setup rather than arithmetic.

Normalization is the clearest example: it must read a token vector, compute statistics, and write a normalized vector. Even if each element needs only a few operations, the bytes still travel. If normalization, a residual addition, and an activation are three separate kernels, one intermediate vector can make several avoidable HBM round trips.

**Kernel fusion** keeps an intermediate tile on-chip while applying several operations in sequence. It changes neither the mathematical model nor its parameter count; it removes traffic between kernels. **FlashAttention**-style IO-aware attention does the same thing for the attention block itself—its win is fewer HBM round trips, not fewer FLOPs.

The scaling rules explain where each term hides:

- dense projections and MLPs scale as `d²`;
- normalization and elementwise work scale as `d`;
- cached attention and cache reads scale as `t·d`.

So elementwise work occupies a larger _fraction_ of latency in narrow models, while long-context cache traffic reappears as a bottleneck even in wide ones—exactly the 5.7ms we found in Part 6.1.

kipply cites a 336M-parameter, `d=1024` study where memory-bound intermediate operations made up roughly **43%** of latency. It is tempting to scale width from 1,024 to 8,192 and divide 43% by eight. Do the ratio properly instead. In a two-component model, the original ratio of linear-cost to quadratic-cost work is

$$r=\frac{T_{\text{linear}}}{T_{\text{quadratic}}}=\frac{0.43}{0.57}\approx0.75.$$

Growing `d` eightfold makes the `d²` term grow eight times faster than the `d` term, so

$$
r'\approx\frac{0.75}{8}\approx0.094
\quad\Longrightarrow\quad
\text{new share}\approx\frac{0.094}{1+0.094}\approx8.6\%,
$$

rather than the 5.4% you get by dividing the percentage directly. (kipply's post lands near 5% by taking that shortcut; the ratio treatment is the stricter version of the same argument.) Even 8.6% is directional only: fusion, attention length, tensor shapes, and implementations all change together. The point is that "small" operations shrink with width but never vanish.

> **Carry forward:** Every cost in the model is now accounted for—weights, cache, math, communication, and the traffic FLOPs hide. The only honest way to check all of it is against a real measurement.

---

## Part 10 — Reality check against a 13B FasterTransformer run

> **Question:** If the arithmetic is correct, why do benchmarks report larger times?

Our equations deliberately compute lower bounds. To see the size of the gap, kipply reported a FasterTransformer benchmark for a differently shaped model:

| Quantity         |    Value |
| ---------------- | -------: |
| `d_model`        |    5,120 |
| Layers           |       40 |
| Heads            | 40 × 128 |
| Context          |      512 |
| Generated tokens |       10 |
| Batch size       |        1 |

The same `24d²L` formula from Part 8 gives its dense baseline:

$$
24 \cdot 40 \cdot 5120^2
= 25.17\text{ GFLOPs/token}.
$$

### 10.1 Decode, one GPU

Its six matrices hold `12d²L = 12.58B` parameters, so they occupy 25.17GB in BF16—numerically the same figure as the GFLOP count, because both are two per parameter. With batch size 1, the ideal weight-bandwidth time is

$$\frac{25.17\text{e}9\text{ bytes}}{1.5\text{e}12\text{ bytes/s}} = 16.8\text{ ms}.$$

Reported measurement: **22.0ms per decode step**.

The gap decomposes cleanly. Sustaining about 90% of peak HBM bandwidth raises the weight term to roughly 18.6ms. Profiled intermediate operations—the Part 9 kernels—contributed about 2.2ms. Launches, embeddings, and sampling closed most of the rest.

### 10.2 Decode, two GPUs

The ideal weight floor halves to about **8.4ms**, but communication appears. The reported measurement was **13.5ms**—not half of 22.0ms. Smaller per-GPU tensors reached less effective bandwidth, the intermediate kernels did not shrink, and profiled communication added roughly 1.7ms.

This is the entire tensor-parallel tradeoff in one measurement: weight time falls, and the terms that do not divide across GPUs become a larger share of what remains. Parts 5.4 and 7 predicted precisely this shape.

### 10.3 Prefill, 512 tokens

With 512 positions at once, prefill is compute-bound, as Part 3.2 argued:

$$
\frac{25.17\text{e}9 \cdot 512}{312\text{e}12}
\approx 41.3\text{ ms}.
$$

The reported one-GPU context time was about **63.2ms**. Even at these large matrix-matrix shapes, real kernels did not reach peak tensor-core throughput: the profile observed roughly **72% of peak for an MLP matmul and 54% for an attention projection**. Prefill also performs causal attention and writes the initial KV cache.

### 10.4 The five recurring theory-to-reality gaps

| Workload                 |       Simple lower bound | Reported measurement | What became visible                                           |
| ------------------------ | -----------------------: | -------------------: | ------------------------------------------------------------- |
| 1-GPU decode             |                   16.8ms |               22.0ms | sustained HBM bandwidth, intermediate kernels, fixed overhead |
| 2-GPU decode             | 8.4ms plus communication |               13.5ms | smaller-shard efficiency and collectives                      |
| 1-GPU, 512-token prefill |        41.3ms dense math |               63.2ms | sub-peak matmul efficiency, attention, and cache writes       |

The five gaps, in the order they usually bite:

1. sustained HBM bandwidth is below the spec-sheet peak;
2. intermediate and elementwise kernels are not free;
3. kernel launches, sampling, and synchronization add fixed costs;
4. real collectives have startup, topology, and bandwidth inefficiencies;
5. matmul efficiency depends on exact dimensions and tiling.

The lesson is not that the formulas failed. They correctly identified which resource would saturate and which bottleneck would appear next. **Arithmetic tells you which regime to investigate; profiling gives you the constants** for one model, runtime, and machine.

> **Carry forward:** Every part so far derived one number for one model. The last step is the procedure itself, so you can rerun the whole chain on a model and machine of your own.

---

## Part 11 — The checklist, applied to any model

> **Question:** How do I redo this whole analysis for a different model and machine?

Work in this order; each step feeds the next, exactly as the parts above did.

**1. Weight capacity.**

$$M_{\text{weights}} = P \cdot b_{\text{weight}}$$

Ours: `52e9 × 2 = 104GB`, so at least 3 A100-40GBs, and 4 for clean head division (Part 4).

**2. KV-cache cost per token, then per live token.**

$$
M_{\text{KV/token}}
= 2b_{\text{cache}}n_{\text{layers}}n_{\text{kv-heads}}d_{\text{head}}
$$

Ours: 2MiB/token, so 32GiB for 16,384 live tokens. Multiply by the sum of all live prompt _and_ generated tokens, never by the request count (Part 2.1).

**3. The dense roofline at your achievable batch.**

$$
T_{\text{large matrices}}
\approx
\max\left(
\frac{M_{\text{weights}}}{N\cdot BW_{\text{HBM}}},
\frac{F_{\text{dense/token}}\cdot B}{N\cdot R_{\text{FLOP}}}
\right)
$$

Ours: `max(17.3ms, 0.66ms)` at `B=8`. Compare `B` against `R_FLOP / BW_HBM` (208 here) to know which side of the crossover you are on (Part 3).

**4. Context-dependent attention, in both FLOPs and bytes.**

$$F_{\text{attention/decode}} \approx 4(t+1)d\,n_{\text{layers}}$$

Ours: 4.30 GFLOPs (4% of arithmetic) but 8GiB read per GPU (25% of step time). Long contexts turn a weight-bound decode into a cache-bandwidth-bound decode (Parts 6.1, 8.3).

**5. Tensor-parallel communication.** Estimate the payload `2Bd`, the collective count `2L`, ring volume `2(N-1)/N`, and startup `α` per collective. Then check the link roofline `d/(2(N-1))` against `R_FLOP / BW_link` to see whether more GPUs will still help (Parts 5, 7).

**6. Validate with the real stack.** Measure prefill latency, time to first token, inter-token latency, throughput, HBM use, and tail latency under the request-length distribution you actually expect—then compare against steps 3–5 to find which of the five gaps from Part 10.4 you are paying.

---

## TL;DR — The whole story in one pass {#tldr}

1. **A request begins with prefill and continues with decode.** Prefill processes the known prompt in parallel and creates the attention state; decode then generates one token per active conversation per step. Future tokens of one response cannot be generated simultaneously.

2. **The KV cache exists because K and V are reused while Q is not.** A token's query serves only its own attention output, but later tokens repeatedly need its key and value, and causal masking makes those immutable once computed. In this model the cache costs `4·L·d`, or **2MiB per live token**.

3. **The cache buys an enormous amount of compute.** K/V re-projection alone is 17.18 GFLOPs per old token—exactly one-sixth of the dense baseline. For our eight conversations, a cacheless step would need about 1,354ms of ideal math instead of 0.66ms. It does not make context free: the `O(t)` attention scan remains, totaling `Θ(nS+n²)` over a generation.

4. **Our workload needs 32GiB of cache and 104GB of weights.** Each 2,048-token conversation holds 4GiB; every step adds 16MiB. One 40GB A100 cannot even hold the weights, so four A100s give each GPU a 26GB shard.

5. **Small-batch decode waits on weight movement, not math.** On four A100s the weight shards need 17.3ms while `B=8` arithmetic needs 0.66ms. Add 5.7ms of KV-cache reads and the memory floor is 23.0ms—about 348 aggregate tokens/s, falling to 332 if communication is fully exposed, so roughly 41–43 tokens/s per conversation. Prefill is the mirror image: 169ms of compute against the same 17.3ms of weight traffic.

6. **Batching reuses one weight stream across more positions.** Arithmetic intensity is about `B` FLOPs/byte, and the A100's balance is `312e12/1.5e12 ≈ 208` FLOPs/byte, so the ideal crossover is near `B=208`. Below it, extra positions mostly fill idle arithmetic capacity; above it, they lengthen the step.

7. **Cache capacity is what caps the batch.** After the weights, four GPUs have ~56GB left, about 26,700 cached tokens—only 13 of our 2,048-token conversations, far below 208. Cutting to 8 KV heads (GQA) shrinks the cache eightfold to 256KiB/token and lifts that ceiling to ~104 conversations, which is why real serving models use fewer KV heads.

8. **Tensor parallelism fixes capacity and weight bandwidth but adds collectives.** Four GPUs cut the weight floor from 69.3ms to 17.3ms, at the cost of two logical all-reduces per block—128 per step. Small batches pay startup (~1.0ms); large batches also pay volume (~5.24ms at `B=500`).

9. **A link roofline says when more GPUs stop helping.** The A100's compute-to-link ratio is ~1040 FLOPs/byte, while attention's output projection achieves `d/(2(N-1))`: 1,365 at `N=4`, 585 at `N=8`, 273 at `N=16`. Per-GPU work shrinks; the activation exchange does not.

10. **Dense FLOPs per token ≈ 2 × dense parameters.** Six matrices per block give `24d²`, so 64 blocks cost 103.08 GFLOPs/token against 51.54B parameters. Left out: cached attention (4.30 GFLOPs at 2,048 tokens), the vocabulary head (0.823 GFLOPs), and all the small kernels whose cost is bytes rather than FLOPs.

11. **Benchmarks are slower for five predictable reasons**—sub-peak bandwidth, intermediate kernels, fixed overhead, real collectives, and shape-dependent matmul efficiency. A 13B run with a 16.8ms lower bound measured 22.0ms, and its two-GPU step fell only to 13.5ms rather than half. Use arithmetic to classify the bottleneck, profiling to get the constants.

For another model or machine, ask these questions in order:

1. Do the weights fit?
2. How many live tokens fit after the weights?
3. At the resulting batch size, is weight movement or arithmetic slower?
4. What communication, cache, and software costs remain exposed?
5. What does the real benchmark say?

---

## References

- [kipply, "Transformer Inference Arithmetic"](https://kipp.ly/p/transformer-inference-arithmetic)
- [kipply, "Transformer Parameter Counting"](https://kipp.ly/p/transformer-param-count)
- [NVIDIA A100 Tensor Core GPU architecture](https://www.nvidia.com/en-us/data-center/a100/)
- [NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) — tensor-parallel communication structure
- [Ivanov et al., "Data Movement Is All You Need"](https://arxiv.org/abs/2007.00072)
- [Dao et al., "FlashAttention"](https://arxiv.org/abs/2205.14135)
