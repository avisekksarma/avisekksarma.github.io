---
layout: post
title: "Transformer Inference Arithmetic, Part 1: Memory and Batching"
subtitle: "KV cache, weight streaming, and why a 52B model on four A100s still waits on memory."
date: 2026-09-16
categories: [Tech, machine-learning, llm]
tags: [machine-learning, llm, systems, transformers, inference, gpu]
series: "Transformer Inference Arithmetic"
series_part: 1
reading_time: 24
description: "A decode step streams the model weights for about one new token and rereads the KV cache. At small batches the GPU waits on memory, not math. This walkthrough uses a 52B model on four A100s to count that cost and see how many conversations still fit after the weights."
featured: true
---

These are notes on [kipply's Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic). I rebuilt them around one 52B model and one workload, so each number comes from the same example instead of a new one every section.

This is part 1 of 2.

- **Part 1 (this post):** what happens for one next token, why the KV cache exists, why decode often waits on memory rather than math, and how many conversations fit after the weights.
- **[Part 2](/blog/2026/transformer-inference-arithmetic-part-2/):** splitting the model across GPUs, putting a full step's latency together, when extra GPUs stop helping, where `FLOPs ≈ 2 × parameters` comes from, and why a real benchmark is slower than the formulas.

[Jump to the recap.](#recap)

---

## The running example

Eight people are chatting with the same model. Each already sent a 2,048-token prompt. The server is generating the next token for all eight conversations at once.

Four jobs show up immediately:

1. keep the model's weights in GPU memory
2. remember attention state for every live token
3. move those weights to the compute units on every decode step
4. combine partial results if the model is split across GPUs

This post covers 1 to 3. Part 2 covers 4.

The chain is simple. Model size decides whether the weights fit. Memory left after the weights decides how many conversations can stay active. Those conversations are the batch. The batch decides whether weight movement or arithmetic is slower.

A few terms, then the numbers.

A **token** is a text piece the model understands: a word, part of a word, punctuation. A **vector** here is 8,192 numbers representing one token. A **parameter** (or weight) is one learned number; most of them sit in matrices. A **FLOP** is one floating-point operation; a multiply plus an add is about two FLOPs. **Prefill** reads the known prompt. **Decode** then writes new tokens one step at a time. The **KV cache** stores attention keys and values from earlier tokens so decode does not rebuild them. A **batch** is token positions processed together; during decode, `B=8` means eight conversations each producing one current token. **HBM** is the GPU's large attached memory. **BF16** is the two-byte number format used for weights and cache in this example.

| Symbol            | Meaning                                   | Value here |
| ----------------- | ----------------------------------------- | ---------: |
| `d` or `d_model`  | width of one token vector                 |      8,192 |
| `L` or `n_layers` | Transformer blocks                        |         64 |
| `n_heads`         | attention heads per block                 |         64 |
| `d_head`          | width of one head                         |        128 |
| `V`               | vocabulary size                           |     50,257 |
| `b`               | bytes per stored number                   |          2 |
| `S`               | prompt length                             |      2,048 |
| `t`               | earlier tokens visible in one decode step |      2,048 |
| `B`               | token positions processed together        |          8 |
| `N`               | GPUs sharing the model                    |          4 |
| `P`               | parameter count                           |        52B |

| Quantity                         |                                          Value |
| -------------------------------- | ---------------------------------------------: |
| Parameters                       |                                     52 billion |
| Token-vector width, `d_model`    |                                          8,192 |
| Transformer blocks               |                                             64 |
| Attention                        | 64 query heads and 64 KV heads, each width 128 |
| MLP hidden width                 |                                  `4d = 32,768` |
| Vocabulary                       |                                         50,257 |
| Weight and cache format          |                        BF16, 2 bytes per value |
| GPU                              |                               NVIDIA A100 40GB |
| Peak BF16 compute                |                           312 trillion FLOPs/s |
| HBM bandwidth                    |                           1.5 trillion bytes/s |
| One-direction GPU-link bandwidth |                            300 billion bytes/s |

Workload:

- 4 A100 GPUs sharing the model
- 8 active conversations
- 2,048 cached tokens per conversation
- so `B=8` on a decode step

This is a teaching setup, not a claim that it is the best production config. Peak compute and bandwidth are upper limits on the hardware, so times computed from them are optimistic lower bounds.

GB is decimal (powers of 1,000). GiB and MiB are binary (powers of 1,024). The labels stay explicit even when the difference is small.

One working rule for this post, derived properly in part 2: if a weight is used once in a matmul, that is about two FLOPs. The six large matrices in this model hold about 51.54B parameters, so they cost about **103.08 billion FLOPs per token**.

---

## From a prompt to one next token

Suppose a user writes:

> The capital of France is

The model never sees the text. The path is:

1. **Tokenize.** Split into pieces and replace each piece with an integer ID.
2. **Embed.** Look up an 8,192-number vector for each ID.
3. **Add position.** Word order matters, so the model needs to know where each token sits.
4. **Run 64 Transformer blocks.** Each block updates the token vectors.
5. **Score the vocabulary.** Turn the last vector into 50,257 scores, one per possible next token.
6. **Pick one token.** Append it and repeat.

<img src="/assets/img/transformer-param-count/01-architecture.svg" alt="Decoder-only Transformer: tokenization and embedding, repeated attention and MLP blocks, then a vocabulary head." style="width: 100%; max-width: 54rem; height: auto; display: block; margin: 1.5rem auto;" />

For an `S`-token prompt the model carries an `S × 8,192` grid through the blocks. Each block keeps that outer shape. Attention gathers information from other positions. The MLP transforms each position on its own.

**Normalization** rescales a vector into a stable range. A **residual addition** adds a block's update to the incoming vector instead of replacing it. Both let information travel through 64 blocks without changing the 8,192-wide shape.

### Text becomes IDs, then vectors

IDs are addresses, not meanings. ID 42 is not "twice as meaningful" as ID 21.

An embedding table `E` of shape `V × d` maps each ID to a learned row:

$$x_i = E[\text{id}_i].$$

Stack those rows for an `S`-token prompt and you get

$$X \in \mathbb{R}^{S \times d} = \mathbb{R}^{S \times 8192}.$$

Without position, the same words in any order would look like an unordered set. Some models add learned position vectors. Many newer ones apply **RoPE** while building queries and keys. Either way, order gets into the representation.

The residual stream stays `[S, d]` all the way through the stack. Only the **LM head** changes the width from `d` to `V`, because only there do we need one score per possible next token. During decode the server usually needs those scores for the newest position, not a fresh prediction from every old one.

### Inside one block

A modern pre-normalized block is roughly:

$$
\begin{aligned}
u &= x + \operatorname{Attention}(\operatorname{LayerNorm}(x)),\\
y &= u + \operatorname{MLP}(\operatorname{LayerNorm}(u)).
\end{aligned}
$$

In order: normalize, attend, add the update back, normalize again, run the MLP, add that update back too.

The six large matrices in this architecture are:

- attention: `Wq`, `Wk`, `Wv`, `Wo`, each `d × d`
- MLP: `W1` of shape `d × 4d`, and `W2` of shape `4d × d`

Attention mixes information **between** positions. The MLP transforms each position **on its own**. Normalization, residuals, masking, activations, and softmax also run. They have far fewer parameters, but they still move data, so they are not free in wall-clock time. Part 2 comes back to that.

### Why attention creates Q, K, and V

Each block projects the token vectors three ways:

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

A projection here just means multiplying by a learned matrix.

A workable picture:

- **query:** what is this position looking for?
- **key:** what does this position advertise about itself?
- **value:** what content can it contribute if selected?

`q · k` does **not** produce `v`. It produces one scalar relevance score. The value vector was computed separately, using `Wv`.

For one current position and one head:

1. take the current query `q`
2. dot it with every allowed key `k_i`
3. divide by `√d_head` so scores do not grow with head width and push softmax into a near one-hot choice
4. mask future positions
5. softmax the scores into weights that sum to one
6. take the weighted sum of the `v_i`s

$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}}+M_{\text{causal}}\right),\qquad Z=AV.$$

This model runs that in 64 **heads**. Each head takes the full 8,192-wide stream and produces a 128-wide Q/K/V slice. Heads split the **projected output width**, not the original input features. Their outputs concatenate back to 8,192 and go through `Wo`. That is why 64 heads do not multiply the FLOP count by 64. Part 2 shows the count.

### Prefill vs decode

This split is most of inference arithmetic.

**Prefill.** All `S` prompt tokens are known. For one head, `Q, K, V` are `S × d_head` and `QK^T` is `S × S`. The causal mask hides the upper triangle, but the prompt can still run in large parallel kernels. Prefill also writes K and V for every prompt position.

**Decode.** Only one new token per request is known. If `t` earlier positions are already cached, the new query is `1 × d_head` and the cache is `t × d_head`. The block computes K and V for the current position, attends over `t+1` entries (it is allowed to see itself), then saves the new K/V. The scores from that position predict the **next** token, which starts the following step.

One conversation cannot write its future tokens in parallel. Token 101 depends on token 100. The eight conversations can advance together, though. Each contributes one current position, so the decode batch is `B=8`.

Prefill is parallel inside one request. Decode is sequential inside a request and parallel across requests. And every later decode step needs those earlier K and V vectors again.

---

## The KV cache

For a token at position `i`, its query is used while computing position `i`'s attention output. A later position `t` makes its own query, so it has no use for `q_i`.

That later position **does** need `k_i` and `v_i`. It compares `q_t` with `k_i`, then uses the score to decide how much of `v_i` to take. Every future position may do that again.

Causality makes storage safe. At a given layer, position `i` cannot see positions after `i`, so a future token cannot change the K or V already computed for `i`. For that request and layer, those vectors do not move.

<img src="/assets/img/transformer-inference-arithmetic/01-kv-cache.svg" alt="A token's query is used once and discarded, while its key and value persist in the cache and are reused by future queries." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

The cache is memoization. Keep an intermediate result because later steps will ask for the same result. It does not change the model's answer.

### Bytes per token

Ordinary multi-head attention stores one K and one V at every layer:

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

The factors:

- two vectors, K and V
- two bytes per BF16 value
- 64 layers, each with its own attention state
- `64 × 128 = 8,192` values in a full K or V vector

Because `n_heads · d_head = d_model`, the BF16 shortcut is

$$\boxed{\text{KV bytes/token} = 4\,n_{\text{layers}}d_{\text{model}}.}$$

One 2,048-token conversation needs

$$2\text{ MiB/token} \times 2048 = 4\text{ GiB}.$$

Eight of them need

$$8\times4\text{ GiB}=32\text{ GiB}.$$

Every decode step adds one cached token per conversation, so another

$$8\times2\text{ MiB}=16\text{ MiB per step}.$$

A memory manager has to count **live tokens**, not requests:

$$M_{\text{KV,total}}=M_{\text{KV/token}}\sum_r t_r.$$

Eight short chats and eight long chats are completely different cache costs. Under tensor parallelism the cache is usually sharded by KV heads. The 32GiB above is the total across the group.

### How much compute the cache saves

For one token at one layer, `Wk` and `Wv` are each `d × d`. One of those matmuls is about `2d²` FLOPs, so both across all layers cost

$$
4 \cdot 64 \cdot 8192^2
= 17{,}179{,}869{,}184
$$

or about **17.18 GFLOPs per old token**. That is exactly one-sixth of the 103.08 GFLOP dense baseline (four of the twenty-four `d²` in `24d²` per block).

Without a cache, every decode step would re-project K and V for all 2,048 earlier positions in all eight conversations:

$$17.18\text{ GFLOPs}\times2048\times8\approx281.5\text{ TFLOPs}.$$

On four ideal A100s that is already

$$\frac{281.5\text{e}12}{4\times312\text{e}12}\approx225.5\text{ ms}.$$

And K/V projections are only one-sixth of the work. A cacheless run would redo the whole dense stack for every position it still needs:

$$\frac{103.08\text{ GFLOPs}\times 8\times2049}{4\times312\text{e}12}\approx1{,}354\text{ ms}.$$

A cached `B=8` step needs about 0.66ms of ideal arithmetic, which we will compute in the next section. The cache turns roughly 1.35 seconds of repeated math into under a millisecond, around a 2,000× cut, in exchange for the 32GiB we just counted. That trade is why the rest of serving is mostly about memory.

What the cache does **not** remove: the current token still computes its own K and V, still reads old K/V, still scores against `t+1` keys, and still sums `t+1` values.

### The cache is not linear

People sometimes say a KV cache turns generation from quadratic into linear. That is almost, but not quite.

- Without a cache, each decode step reruns projections over the whole prefix.
- With a cache, the big projections and the MLP run only for the new token.
- The new query still scans `t` cached keys and values.

So the **attention scan per decode step is still `O(t)`**. For a prompt of length `S` and then `n` decode steps, the prior-context scans total

$$nS+\frac{n(n-1)}{2}=\Theta(nS+n^2).$$

One of our conversations generating 500 tokens after a 2,048-token prompt scans

$$500\times2048+\frac{500\times499}{2}=1{,}148{,}750$$

key/value positions per layer. That still grows with output length squared, even with a perfect cache.

The win is that the expensive `d²` projection and MLP work no longer repeats for old positions. Only the `t·d` scan grows. For a wide model at ordinary context lengths, dropping repeated `d²` work is huge even though the sequence-length complexity is not linear. At long contexts the leftover `t·d` term, and the bytes it reads, show up again as time.

### GQA: fewer KV heads, smaller cache

Our model stores 64 KV heads, one per query head. Grouped-query and multi-query attention let several query heads share one K/V pair:

$$\text{KV bytes/token} = 2b\,n_{\text{layers}}n_{\text{kv-heads}}d_{\text{head}}.$$

If this 52B model kept 64 query heads but only 8 KV heads, each token would cost

$$2\cdot2\cdot64\cdot8\cdot128=262{,}144\text{ bytes}=256\text{ KiB},$$

one-eighth of 2MiB. The eight conversations would then hold **4GiB instead of 32GiB**. That gap comes back later as 13 conversations vs about 104, and as 5.7ms of cache reads vs 0.7ms. Serving models almost always cut KV heads for this reason.

The cache solved a compute problem by creating a memory problem. We already need 32GiB for eight conversations, and we have not counted the weights yet.

---

## Why decode waits on memory, not math

An A100 can do 312 trillion operations per second. Generating one token can still take many milliseconds. "Load the model" hides two different moves.

**Storage to HBM.** At startup, the checkpoint goes from disk or host memory into GPU HBM. That can take seconds. It happens once, before serving, so it is not per-token latency.

**HBM to the compute units.** On every forward pass, kernels fetch small weight tiles into on-chip caches, shared memory, and registers. The checkpoint stays in HBM. The GPU cannot keep a 104GB model next to the arithmetic units.

<img src="/assets/img/transformer-inference-arithmetic/04-hardware-data-path.svg" alt="Model weights move from storage to GPU HBM once at startup, then weight tiles stream from HBM through small on-chip memory to compute units on every forward pass." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

_[Open the hardware data-path diagram at full size](/assets/img/transformer-inference-arithmetic/04-hardware-data-path.svg)._

HBM is a warehouse next to a factory. Startup stocks the warehouse once. Every decode step still brings pallets to the factory floor. The weights are on the GPU. They are not all sitting beside the ALUs.

Tiles get reused while a kernel runs, and a little stays in cache, so "stream the weights" is a first-order picture of the traffic. The model is far larger than on-chip storage, so a decode pass still fetches roughly one model's worth of dense weights from HBM.

For 52B BF16 parameters:

$$52\text{e}9\times2\text{ bytes}=104\text{ GB}.$$

On a made-up single A100 with enough capacity, reading those bytes once would take at least

$$T_{\text{weights,1}} = \frac{104\text{e}9}{1.5\text{e}12} = 69.3\text{ ms}.$$

The model does not fit on one 40GB A100. Across four GPUs, each owns and reads about 26GB:

$$T_{\text{weights,4}} = \frac{26\text{e}9}{1.5\text{e}12} = 17.3\text{ ms}.$$

Compare that with arithmetic. 103.08 billion FLOPs per token, split across four ideal A100s:

$$T_{\text{math,token}}=\frac{103.08\text{e}9}{4\times312\text{e}12}\approx0.083\text{ ms}.$$

For our batch of eight:

$$T_{\text{math},B=8}=8\times0.083\approx0.66\text{ ms}.$$

0.66ms of math, 17.3ms of weight traffic. About a factor of 26. The arithmetic units sit idle waiting for data. That is what batching is for.

### Why batching helps

One BF16 weight costs two bytes to read. Using it for one token is about one multiply and one add: two FLOPs. That is about **1 FLOP per byte**, well below what the A100 can sustain.

If `B` positions share the same matrix, the matrix is fetched once and every weight is reused across `B` rows. Weight traffic stays about the same. Useful arithmetic grows with `B`. With `P` weights in the matrix:

$$
\text{arithmetic intensity} \approx
\frac{2BP\text{ FLOPs}}{2P\text{ bytes}}
= B\text{ FLOPs/byte}.
$$

Arithmetic intensity is useful operations per byte moved. A batch of 8 gets about eight uses from each fetched weight. A batch of 208 gets about 208.

The A100's own compute-to-bandwidth ratio is

$$\frac{312\text{e}12}{1.5\text{e}12} \approx 208\text{ FLOPs/byte}.$$

Match the workload's `B` FLOPs/byte with the hardware's 208 FLOPs/byte and the ideal **crossover** sits near `B=208`:

- **Below 208:** weights cannot arrive fast enough to keep the ALUs busy. Memory-bandwidth-bound.
- **Near 208:** weight delivery and arithmetic take similar time.
- **Above 208:** the ALUs are full, so extra positions make the step longer. Compute-bound.

During decode, `B` is concurrent conversations each contributing one current token. It is not one conversation writing 208 future tokens at once.

<img src="/assets/img/transformer-inference-arithmetic/02-batching-crossover.svg" alt="Ideal batching roofline for the 52B model on four A100 GPUs: a 17.3 millisecond weight-streaming floor meets the compute line near a batch of 208." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

_[Open the batching crossover graph at full size](/assets/img/transformer-inference-arithmetic/02-batching-crossover.svg)._

`TP=4` means four-way tensor parallelism. Using only the six large matrices:

| Positions processed together (`B`) | Weight floor, TP=4 | Ideal math time, TP=4 | Large-matrix lower bound |
| ---------------------------------: | -----------------: | --------------------: | -----------------------: |
|                                  1 |            17.3 ms |              0.083 ms |                  17.3 ms |
|                       **8 (ours)** |        **17.3 ms** |           **0.66 ms** |              **17.3 ms** |
|                                 50 |            17.3 ms |               4.13 ms |                  17.3 ms |
|                                100 |            17.3 ms |               8.26 ms |                  17.3 ms |
|                            **208** |        **17.3 ms** |           **17.2 ms** |             **≈17.3 ms** |
|                                500 |            17.3 ms |               41.3 ms |                  41.3 ms |

The last column is

$$T_{\text{large matrices}}\approx\max(T_{\text{weight read}},T_{\text{math}}),$$

because a good matmul kernel computes on one tile while fetching the next. The slower stream sets the floor. Extra positions below 208 are not free. They use arithmetic that would otherwise sit idle while weights arrive. Cache reads, attention, communication, and software give the "flat" line a real slope.

Two notes on the table. The 17.3 vs 17.2 blip at `B=208` is the rounded 52B weight footprint on the memory line vs the 51.54B block-matrix count on the math line. Use one consistent weight set and the crossover lands on 208 exactly. And the `B=500` row is hypothetical: four A100s cannot hold 500 of our 2,048-token conversations, which is the next section.

### Prefill is the opposite

Prefill has `B = S = 2048` positions at once, so the same formulas put it in a different regime.

Dense math to prefill one 2,048-token prompt on four A100s:

$$\frac{103.08\text{e}9\times2048}{4\times312\text{e}12}\approx169\text{ ms},$$

against the same 17.3ms weight floor. Prefill is about **ten times compute-bound**. Our decode step was 26 times memory-bound.

Prefill attention is real but smaller. The `S × S` score matrices across all layers cost

$$4S^2dL=4\cdot2048^2\cdot8192\cdot64\approx8.8\text{ TFLOPs}$$

(about half of that once you exploit the causal mask), roughly 4% of the dense prefill work, or 3.5 to 7ms.

That is why serving systems report two latencies. **Time to first token** is mostly compute-bound prefill, about 169ms here before extras. **Inter-token latency** is mostly memory-bound decode. Cache reads add another 5.7ms on top of the 17.3ms weight floor (32GiB of K/V, 8GiB per GPU, 1.5TB/s), so the memory floor is already about 23ms before communication. Part 2 puts the rest of that step together.

Our `B=8` decode is far below the 208-position crossover. More conversations would fill idle arithmetic almost for free. Each extra conversation needs KV-cache memory.

---

## Does it even fit?

Weights alone need at least

$$\left\lceil\frac{104}{40}\right\rceil = 3\text{ A100-40GB GPUs}.$$

"The weights fit" is not enough. Whatever is left has to hold the KV cache and the runtime's temporary buffers. Remainder divided by 2,097,152 bytes per token:

| GPUs | Total HBM | Weight memory | Theoretical remainder | Theoretical KV-token capacity |
| ---: | --------: | ------------: | --------------------: | ----------------------------: |
|    3 |    120 GB |        104 GB |                 16 GB |                 ≈7,629 tokens |
|    4 |    160 GB |        104 GB |                 56 GB |                ≈26,703 tokens |

Our eight conversations already have

$$8\times2{,}048=16{,}384\text{ live tokens}.$$

Three GPUs can hold the weights and **cannot** hold this cache. Four GPUs can. The cache uses 32GiB (about 34.4GB decimal), leaving roughly 21.6GB before runtime overhead.

Three is only the capacity floor, and it is awkward for another reason. 64 attention heads divide cleanly across 4, 8, or 16 GPUs, not 3, and most GPU servers have power-of-two link topologies. Four is the natural count here.

### Capacity is a throughput limit

The 26,703-token ceiling limits the batch. If every conversation is 2,048 tokens long, four GPUs can theoretically keep only

$$\left\lfloor\frac{26{,}703}{2{,}048}\right\rfloor=13\text{ such conversations}.$$

A decode batch near 13 is nowhere near the 208-position crossover. Long contexts eat the memory that would have bought concurrency, and concurrency is what makes weight streaming efficient. Capacity decides how many requests even exist to batch.

13 is also a math ceiling, not a safe scheduler limit. A real runtime still needs:

- temporary activations and kernel workspaces
- communication buffers
- allocator headroom and fragmentation
- CUDA context and library allocations
- possibly a separate vocabulary-head matrix

**Paged KV-cache allocators** cut fragmentation by handing out cache in fixed blocks instead of one huge contiguous region per request. They use the budget better. They cannot exceed the physical bytes.

Now the GQA variant. Hold the weights at 104GB. The same 56GB remainder at 256KiB per token instead of 2MiB holds

$$
\frac{56\text{e}9}{262{,}144}\approx213{,}600\text{ tokens}
\quad\Longrightarrow\quad
\frac{213{,}600}{2048}\approx104\text{ conversations}.
$$

Same hardware, one architecture choice, and the reachable decode batch moves from 13 to about 104, next to the 208 crossover.

104 is a conservative floor, because fewer KV heads also shrink the weights. With 8 KV heads, `Wk` and `Wv` become `8192 × 1024` instead of `8192 × 8192`, which drops about 7.52B parameters (15.0GB). That would leave about 71GB for cache and push the ceiling past 130 conversations. Real designs usually spend those freed parameters somewhere else (more layers, a wider MLP), so holding the model at 52B is the fairer comparison.

Four GPUs solve the capacity problem. No single GPU now computes the full answer, so they have to exchange partial results. That is [part 2](/blog/2026/transformer-inference-arithmetic-part-2/).

---

## Recap {#recap}

1. **Prefill, then decode.** Prefill reads the prompt in parallel and writes the cache. Decode writes one token per active conversation per step. One response cannot generate its own future tokens at once.

2. **Q is used once. K and V are reused.** Causal masking makes those K/V vectors fixed once computed. In this model the cache costs `4·L·d`, which is **2MiB per live token**.

3. **The cache buys a lot of compute.** K/V re-projection is 17.18 GFLOPs per old token, one-sixth of the dense baseline. For our eight conversations a cacheless step would need about 1,354ms of ideal math instead of 0.66ms. Context is not free: the `O(t)` attention scan remains, totaling `Θ(nS+n²)` over a generation.

4. **32GiB of cache, 104GB of weights.** Each 2,048-token conversation holds 4GiB. Every step adds 16MiB. One 40GB A100 cannot hold the weights. Four A100s give each GPU a 26GB shard.

5. **Small-batch decode waits on weight movement.** On four A100s the shards take 17.3ms and `B=8` math takes 0.66ms. Cache reads add about 5.7ms, so the memory floor is already ~23ms before communication. Prefill is the mirror image: 169ms of compute against the same 17.3ms of weight traffic.

6. **Batching reuses one weight stream.** Intensity is about `B` FLOPs/byte. The A100's balance is 208 FLOPs/byte, so the crossover is near `B=208`. Below that, extra positions mostly fill idle math. Above it, they lengthen the step.

7. **Cache capacity caps the batch.** After the weights, four GPUs have about 56GB left, ~26,700 cached tokens, only 13 of our 2,048-token conversations. Eight KV heads (GQA) shrink the cache to 256KiB/token and lift that ceiling to about 104 conversations.

**Next:** [Part 2, Tensor Parallelism and Real Latency](/blog/2026/transformer-inference-arithmetic-part-2/). Four GPUs cut the weight floor from 69.3ms to 17.3ms, add 128 logical all-reduces per step, and still do not match a spec-sheet benchmark.

---

## References

- [kipply, "Transformer Inference Arithmetic"](https://kipp.ly/p/transformer-inference-arithmetic)
- [kipply, "Transformer Parameter Counting"](https://kipp.ly/p/transformer-param-count)
- [NVIDIA A100 Tensor Core GPU architecture](https://www.nvidia.com/en-us/data-center/a100/)
- [Counting Transformer Parameters](/blog/2026/counting-transformer-parameters/) (the `12d²` count this series starts from)
