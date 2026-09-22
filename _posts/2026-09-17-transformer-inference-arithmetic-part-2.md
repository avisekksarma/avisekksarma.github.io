---
layout: post
title: "Transformer Inference Arithmetic, Part 2: Tensor Parallelism and Real Latency"
subtitle: "Splitting a 52B model across four A100s, when extra GPUs stop helping, and why a benchmark is slower than the formulas."
date: 2026-09-17
categories: [Tech, machine-learning, llm]
tags: [machine-learning, llm, systems, transformers, inference, gpu]
series: "Transformer Inference Arithmetic"
series_part: 2
reading_time: 26
description: "Once a model is split across GPUs, each decode step also has to combine partial results. Extra GPUs stop helping once communication, not compute, is the floor."
featured: true
---

[Part 1](/blog/2026/transformer-inference-arithmetic-part-1/) got us to a cliff. A 52B BF16 model needs 104GB of weights. The KV cache for eight 2,048-token conversations needs another 32GiB. Three 40GB A100s can hold the weights and cannot hold that cache. Four GPUs can. No single GPU now owns the full answer.

This post is what happens after you split.

Same model, same eight conversations, same four A100s. Peak rates are still upper limits, so the times below are still optimistic lower bounds.

[Jump to the recap.](#recap)

---

## Where part 1 left the numbers

| Quantity                        |          Value |
| ------------------------------- | -------------: |
| Weights                         |         104 GB |
| Cache, 8 × 2,048 tokens         |         32 GiB |
| Weight-read floor, 4 GPUs       |        17.3 ms |
| Ideal math, `B=8`               |        0.66 ms |
| KV-cache read, 4 GPUs           |         5.7 ms |
| Memory floor so far             |        23.0 ms |
| Dense FLOPs per token           | 103.08 billion |
| Ideal crossover                 |      `B ≈ 208` |
| Conversations that actually fit |            ~13 |

The 103.08 billion FLOP figure has been a working rule so far (`FLOPs ≈ 2 × parameters` on the six large matrices). This post derives it, then checks the whole picture against a published 13B run.

A **kernel** is one GPU program, a matmul or a normalization or similar. **Tensor parallelism** splits one model across several GPUs. Each GPU computes a piece, then some pieces have to be combined. An **all-reduce** is that combine: exchange and sum partial answers.

---

## Splitting weights across GPUs

Tensor parallelism shards each large matrix. With four GPUs, each package owns about a quarter of the weight bytes and does about a quarter of the matmul work:

$$104\text{ GB}/4=26\text{ GB of weights per GPU}.$$

Each GPU has its own HBM channels and compute units, so this is more than pooling capacity. Four devices can stream four shards and multiply them in parallel. That is how part 1's 69.3ms single-device weight floor became 17.3ms.

<img src="/assets/img/transformer-inference-arithmetic/03-tensor-parallel.svg" alt="Four tensor-parallel GPUs each read a 26GB weight shard, compute a partial output, and exchange partials in a collective to reconstruct the full activation." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

### Why they have to talk

Imagine splitting a long sum among four people. Each can compute a quarter on their own. Nobody has the total until the four partial sums are combined. Sharded matrix multiplication has the same dependency.

Depending on which matrix dimension you partition, each GPU produces either:

- a distinct slice of the output that you can concatenate, or
- a **partial sum** that has to be added across GPUs

The combine is an **all-reduce**. A Megatron-style block is arranged so that only **two logical all-reduces per block** are needed:

1. after attention's output projection `Wo`
2. after the MLP's down-projection `W2`

The other four matrices skip an immediate sync. `Wq`, `Wk`, `Wv`, and the MLP up-projection `W1` are column-parallel: each GPU produces its own slice of heads or hidden units and keeps working on that slice locally. `Wo` and `W2` are row-parallel. They take a sharded input and produce a full-width residual-stream update, so their partial sums have to be combined before the next operation reads that stream.

With 64 blocks, one decode step does

$$2\times64=128\text{ logical all-reduces}.$$

Each all-reduce is usually two network phases, reduce-scatter then all-gather, which is why some writeups count four communication phases per block. Calling those phases "four all-reduces" overcounts the sync points.

That sounds expensive. Communication has two separate costs, and which one wins depends on batch size.

### Small batches mostly pay startup

Starting a collective has a fixed cost even when the payload is tiny, like placing a phone call before you say anything.

For our `B=8` workload, one **activation payload** (the current `B × d` slice of the residual stream) is only

$$M_{\text{payload}}=2Bd=2\times8\times8192=131{,}072\text{ bytes}=128\text{ KiB}.$$

If one collective has an optimistic `8μs` startup, 128 of them contribute about

$$128\times8\mu\text{s}\approx1.0\text{ ms}.$$

At this batch, startup matters more than the bytes. Two caveats. The `8μs` is an assumption, not a spec; real collective latency depends on generation, topology, and the library. And this charges startup once per _logical_ all-reduce. If each of the two network phases pays its own message latency, the term doubles to about `4·64·8μs ≈ 2.0ms`, which is how kipply writes it.

### Large batches also pay for bytes

The `B=500` point on part 1's graph is reachable only with much shorter contexts or a lot more memory. One payload there grows to

$$2\times500\times8192=8.192\text{ MB}.$$

For a ring all-reduce over `N` GPUs, a common estimate for bytes **sent per GPU** is

$$M_{\text{ring}} \approx 2\frac{N-1}{N}M_{\text{payload}}.$$

The factor `2(N-1)/N` covers reduce-scatter plus all-gather. Each GPU receives the same amount at the same time on a full-duplex link. The factor approaches 2 as the group grows. At `N=4` that is about **12.288MB per logical all-reduce**, so 128 collectives move

$$12.288\text{ MB}\times128\approx1.57\text{ GB per GPU},$$

and at an idealized 300GB/s the bandwidth term is

$$T_{\text{comm,volume}} \approx \frac{1.57\text{e}9}{300\text{e}9} \approx 5.24\text{ ms}.$$

Shape of the cost:

- small batch: mostly "start 128 exchanges" (~1.0ms of startup, almost no bytes)
- large batch: startup **plus** moving much larger activations (~5.24ms of volume at `B=500`)

Exact values depend on links, topology, the collective, whether bandwidth is quoted per direction or aggregate, message size, and how much the runtime can overlap.

### More GPUs do not keep cutting latency in half

Eight GPUs would halve each weight shard from 26GB to 13GB, so the ideal weight-read floor falls from 17.3ms to about 8.7ms. The 8,192-wide activation still has to be combined, now among more participants, and startup latency does not shrink.

More GPUs reduce local memory traffic and arithmetic. They do not remove collective startup or make the exchanged activation smaller. At some point communication is large compared with each GPU's shrinking local work. The next section after the latency model puts a number on that point.

A decode step is not "compute time plus every other number." Weight reads and arithmetic overlap. Some communication sits between dependent stages. We need one picture that keeps those relationships straight.

---

## One decode step, all the costs

There is no single "model latency." A serving system usually tracks at least four numbers:

- **prefill latency / time to first token:** how long until generation begins
- **inter-token latency:** time between consecutive output tokens for one request
- **batch step time:** how long one scheduler iteration takes
- **throughput:** total output tokens per second across all requests

A usable step-time split is

$$
T_{\text{step}}
\approx T_{\text{large matrices}}
+ T_{\text{attention and cache}}
+ T_{\text{communication on the critical path}}
+ T_{\text{fixed overhead}},
$$

where the large-matrix term is the roofline from part 1:

$$
T_{\text{large matrices}}
\approx
\max\left(
\frac{\text{weight bytes}}{N\cdot BW_{\text{HBM}}},
\frac{F_{\text{dense/token}}\cdot B}{N\cdot R_{\text{FLOP}}}
\right).
$$

The max inside a matmul is because an optimized kernel computes on one tile while fetching another. Between dependent stages, a required collective can sit on the critical path. The general bound is a sandwich:

$$
\max(T_{\text{compute}},T_{\text{comm}})
\leq T_{\text{compute+comm}}
\leq T_{\text{compute}}+T_{\text{comm}}.
$$

You should not always add every term, and you should not always take one maximum. Real runtimes overlap what they safely can. The truth sits between perfect overlap and full serialization.

### Our eight-conversation step

We already have:

- weight-read floor, TP=4: **17.3ms**
- ideal math for `B=8`: **0.66ms**
- collective startup for 128 all-reduces: **~1.0ms**, volume negligible at 128KiB payloads

So the large-matrix term is about 17.3ms. The 32GiB of cache now shows up as time, not just bytes. During one decode step, attention reads the K/V state for all eight conversations. Sharded evenly, that is 8GiB per GPU. At an ideal 1.5TB/s:

$$T_{\text{KV read}}\approx\frac{8\text{ GiB}}{1.5\text{ TB/s}}\approx5.7\text{ ms}.$$

Weights and cache share the same HBM bandwidth, so together they set a memory-traffic floor of

$$17.3+5.7=23.0\text{ ms}.$$

If the ~1.0ms of collective cost is fully exposed, the estimate is about **24.1ms** before software (about 25.1ms if each collective's two phases each pay startup). With good overlap it stays near 23.0ms. Implementations change the exact traffic. Cache reads at this context length are about a quarter of the step.

The step produces eight next tokens, so the throughput ceiling is

$$\frac{8}{0.0230}\approx348\text{ tokens/s aggregate},$$

or about 332 if communication is fully exposed. That is roughly **41 to 43 tokens/s per continuously active conversation**. Real throughput is lower once software is included.

GQA on this same step: cache reads drop from 5.7ms to 0.7ms, so the memory floor falls from 23.0ms to about 18.0ms even before the smaller `Wk`/`Wv` shrink the weight term, and the reachable batch grows eightfold. Both effects go the same way.

### Bigger batch, more throughput, maybe worse latency

Suppose the requests were much shorter, or the server had more memory, so it could form `B=208`. Weight and math would both be about 17.3ms, and the large-matrix-only ceiling would approach

$$\frac{208}{0.0173}\approx12{,}000\text{ positions/s}$$

before cache, communication, and software. Compare the extremes. One idealized `B=1` step costs about 18.3ms (17.3ms of weight streaming plus ~1.0ms of exposed startup, cache reads left out on both sides), so serving 500 requests one at a time would occupy

$$500\times18.3\text{ ms}=9.15\text{ s}.$$

One `B=500` step is on the order of 47.6ms (41.3ms of math, 5.24ms of ring volume, ~1.0ms of startup), about a **192× gain in aggregate work per unit time**.

That is aggregate work, not per-user speed. A request may wait while the scheduler forms the batch, and past the crossover the step itself gets longer.

- larger batch: better total hardware efficiency
- individual request: maybe more queueing and a longer step

Batching helps throughput only when enough live work exists **and** the cache for that work fits. `B=500` still means 500 requests each producing one token.

Two loose threads. When does communication overtake compute as `N` grows? And where did 103.08 GFLOPs come from?

---

## When communication takes over

Picture each GPU as a factory and the interconnect as a fleet of trucks. A factory computes one partial activation. The trucks exchange those partials so every factory can continue with the combined result. If local compute takes much longer than the transfer, communication is a small tax. If sharding makes each factory's job tiny while the shipment stays activation-sized, the trucks become the limit.

Part 1 compared compute against HBM bandwidth. Now compare compute against _link_ bandwidth:

$$\frac{312\text{e}12\text{ FLOP/s}}{300\text{e}9\text{ byte/s}} \approx 1040\text{ FLOPs/byte}.$$

This is a second roofline. A stage doing fewer than about 1040 useful FLOPs per communicated byte risks becoming communication-bound.

Apply it to attention's row-parallel output projection `Wo`. Each GPU multiplies an input slice of width `d/N` into a full-width partial output, so for `B` positions it does

$$F_{\text{Wo,GPU}}=\frac{2Bd^2}{N}\text{ FLOPs},$$

while the ring all-reduce sends

$$M_{\text{ring}}=2\frac{N-1}{N}(2Bd)=4Bd\frac{N-1}{N}\text{ bytes per GPU}.$$

Local work over bytes sent:

$$I_{\text{Wo,link}}=\frac{F_{\text{Wo,GPU}}}{M_{\text{ring}}}=\frac{d}{2(N-1)}\text{ FLOPs/byte}.$$

`B` cancels. This is a property of the sharding, not something a bigger batch can fix. The MLP down-projection starts from width `4d`, so it does four times as much local math for the same `B × d` payload:

$$I_{\text{W2,link}}=\frac{2d}{N-1}\text{ FLOPs/byte}.$$

For `d=8192`:

| Tensor-parallel size | `Wo` intensity | `W2` intensity | Against 1040 FLOPs/byte                        |
| -------------------: | -------------: | -------------: | ---------------------------------------------- |
|                **4** |      **1,365** |      **5,461** | `Wo` has a narrow theoretical compute cushion  |
|                    8 |            585 |          2,341 | `Wo` communication is exposed                  |
|                   16 |            273 |          1,092 | `Wo` is strongly exposed; `W2` is near balance |

Four GPUs sit just above the line. Eight would push attention's output projection below it. That is the diminishing return from earlier, now with a number attached.

This is not a universal cutoff. Faster links, topology-aware collectives, overlap, quantized communication, and different matrix shapes all move it. The direction is stable:

> More GPUs reduce each GPU's weight traffic and math. The activation being exchanged does not shrink at the same rate, and startup latency does not shrink at all.

The minimum GPU count is a capacity question (part 1). The best GPU count is a latency, throughput, and cost question. This roofline is how you bound it before benchmarking.

Both rooflines, HBM in part 1 and links here, compared arithmetic against bytes, and both used 103.08 GFLOPs per token. Time to earn that number.

---

## Where FLOPs per token ≈ 2 × parameters comes from

### One small matrix

Multiplying an `m × n` matrix by a length-`n` vector does about

$$2mn\text{ FLOPs},$$

one multiply and one add per matrix element. The matrix has `mn` parameters. If a weight is used once in a matmul,

$$\text{FLOPs} \approx 2 \times \text{parameters}.$$

A tiny example. Let

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

`W` has 6 parameters. `Wx` uses each of them once: six multiplies and about six adds, so roughly 12 FLOPs. The exact elementary count is `2mn-m`, because the first term of each sum needs no preceding add. At width 8,192 that correction is tiny, and hardware specs usually count a multiply-accumulate as two FLOPs anyway.

### The six large matrices in one block

| Matrix | Shape          |  Parameters | FLOPs/token |
| ------ | -------------- | ----------: | ----------: |
| `Wq`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wk`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wv`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `Wo`   | `8192 × 8192`  |  67,108,864 | 134,217,728 |
| `W1`   | `8192 × 32768` | 268,435,456 | 536,870,912 |
| `W2`   | `32768 × 8192` | 268,435,456 | 536,870,912 |

Grouped:

| Part of one block | Large matrices                             | Parameters | FLOPs per token |
| ----------------- | ------------------------------------------ | ---------: | --------------: |
| Attention         | Q, K, V, and output: four `d × d` matrices |      `4d²` |           `8d²` |
| MLP               | one `d × 4d` and one `4d × d` matrix       |      `8d²` |          `16d²` |
| **Total**         | six matrices                               | **`12d²`** |      **`24d²`** |

The 64 heads do **not** add another factor of 64. `Wq` is 64 narrower `8192 × 128` projections side by side:

$$64\times8192\times128=8192\times8192=d^2.$$

Heads split the output width. Their widths sum back to `d_model`, the same fact that made the KV-cache formula collapse to `4·L·d` in part 1.

One block therefore costs

$$
4(2d^2) + 2(2d\cdot4d)
= 24d^2
= 1{,}610{,}612{,}736\text{ FLOPs/token}.
$$

Across 64 blocks:

$$
24 \cdot 64 \cdot 8192^2
= 103{,}079{,}215{,}104
\approx 103.08\text{ GFLOPs/token}.
$$

Divide by two and you get the weight count of those matrices:

$$\frac{103.08\text{B}}{2} = 51.54\text{B parameters},$$

almost the entire 52B model. So

$$\boxed{\text{dense FLOPs per token}\approx2\times\text{dense parameters}.}$$

### What the 2P shortcut leaves out

103.08 GFLOPs is the dense projection-plus-MLP baseline, not the whole decode step.

**Cached attention.** With `t` earlier positions plus the current one, QK scores and the weighted-V sum add about

$$4(t+1)d\,n_{\text{layers}}\approx4td\,n_{\text{layers}}\text{ FLOPs}.$$

At our `t=2048`:

$$
4 \cdot 2049 \cdot 8192 \cdot 64
\approx 4.30\text{ GFLOPs},
$$

about **4.2%** of the six-matrix baseline, growing linearly with context on every decode step. Same cached positions were only 4% of the _arithmetic_ in the latency section and about 25% of the _time_, because attention re-reads 8GiB per GPU while doing very little math per byte.

**Vocabulary head.** Projecting the final hidden state to 50,257 logits costs

$$
2Vd
= 2 \cdot 50{,}257 \cdot 8192
\approx 0.823\text{ GFLOPs}.
$$

The embedding table holds about 411.7M parameters. Add them to the block matrices and the model size closes:

$$51.54\text{B}+0.41\text{B}\approx51.95\text{B}\approx52\text{B}.$$

If embeddings are **tied**, those same parameters act as the vocabulary head and _do_ participate in a matmul at output time, even though the input embedding is only a lookup. If untied, the model stores a second matrix of the same size.

**Other kernels.** Normalization, RoPE, residuals, activations, softmax, sampling, and cache reads add smaller FLOP counts and nonzero latency.

`2P` is useful because most large-model parameters are used once per token in a dense matmul. It is a baseline, not an exact law. And we already saw that 4% of the arithmetic ate 25% of the step, so FLOPs alone do not predict time.

---

## Memory traffic that FLOPs miss

FLOP accounting makes normalization, softmax, residuals, and activations look trivial. On a GPU, an operation can do almost no arithmetic and still spend real time reading and writing tensors.

Three effects:

1. **Intermediate activation traffic.** Unfused kernels may write a tensor to HBM only for the next kernel to read it straight back.
2. **KV-cache traffic.** Each decode step reads all earlier K/V that attention needs, our 8GiB per GPU.
3. **Kernel launch and sync.** Tiny kernels can be dominated by setup rather than arithmetic.

Normalization is the cleanest example. It has to read a token vector, compute statistics, and write a normalized vector. Even if each element needs only a few operations, the bytes still travel. If normalization, a residual add, and an activation are three separate kernels, one intermediate vector can make several extra HBM round trips.

**Kernel fusion** keeps an intermediate tile on-chip while applying several operations in sequence. It does not change the math or the parameter count. It removes traffic between kernels. **FlashAttention**-style IO-aware attention does the same thing for the attention block: fewer HBM round trips, not fewer FLOPs.

Where each term hides:

- dense projections and MLPs scale as `d²`
- normalization and elementwise work scale as `d`
- cached attention and cache reads scale as `t·d`

So elementwise work is a larger _fraction_ of latency in narrow models, while long-context cache traffic comes back even in wide ones. That is the 5.7ms from earlier.

kipply cites a 336M-parameter, `d=1024` study where memory-bound intermediate operations made up about **43%** of latency. It is tempting to scale width from 1,024 to 8,192 and divide 43% by eight. Do the ratio instead. In a two-component model,

$$r=\frac{T_{\text{linear}}}{T_{\text{quadratic}}}=\frac{0.43}{0.57}\approx0.75.$$

Growing `d` eightfold makes the `d²` term grow eight times faster than the `d` term, so

$$
r'\approx\frac{0.75}{8}\approx0.094
\quad\Longrightarrow\quad
\text{new share}\approx\frac{0.094}{1+0.094}\approx8.6\%,
$$

rather than the 5.4% you get by dividing the percentage. kipply's post lands near 5% by taking that shortcut. The ratio is the stricter version of the same argument. Even 8.6% is only directional: fusion, attention length, shapes, and implementations all change together. "Small" operations shrink with width. They do not vanish.

Weights, cache, math, communication, and the traffic FLOPs hide are all on the page now. The honest check is a real measurement.

---

## A 13B FasterTransformer run

The equations compute lower bounds. kipply reported a FasterTransformer benchmark on a different shape:

| Quantity         |    Value |
| ---------------- | -------: |
| `d_model`        |    5,120 |
| Layers           |       40 |
| Heads            | 40 × 128 |
| Context          |      512 |
| Generated tokens |       10 |
| Batch size       |        1 |

The same `24d²L` formula gives its dense baseline:

$$
24 \cdot 40 \cdot 5120^2
= 25.17\text{ GFLOPs/token}.
$$

### Decode, one GPU

The six matrices hold `12d²L = 12.58B` parameters, so they occupy 25.17GB in BF16. Same figure as the GFLOP count, because both are two per parameter. With batch size 1, the ideal weight-bandwidth time is

$$\frac{25.17\text{e}9}{1.5\text{e}12} = 16.8\text{ ms}.$$

Reported measurement: **22.0ms per decode step**.

The gap is readable. About 90% of peak HBM bandwidth raises the weight term to roughly 18.6ms. Profiled intermediate operations (the kernels from the previous section) contributed about 2.2ms. Launches, embeddings, and sampling closed most of the rest.

### Decode, two GPUs

The ideal weight floor halves to about **8.4ms**, but communication appears. The reported measurement was **13.5ms**, not half of 22.0ms. Smaller per-GPU tensors reached less effective bandwidth, the intermediate kernels did not shrink, and profiled communication added about 1.7ms.

Weight time falls. The terms that do not divide across GPUs become a larger share of what remains. That is the tensor-parallel tradeoff in one measurement, the same shape the link roofline predicted.

### Prefill, 512 tokens

With 512 positions at once, prefill is compute-bound, as part 1 argued:

$$
\frac{25.17\text{e}9 \cdot 512}{312\text{e}12}
\approx 41.3\text{ ms}.
$$

The reported one-GPU context time was about **63.2ms**. Even at these large matrix-matrix shapes, real kernels did not hit peak tensor-core throughput. The profile saw about **72% of peak for an MLP matmul and 54% for an attention projection**. Prefill also does causal attention and writes the initial KV cache.

### Five recurring gaps

| Workload                 |       Simple lower bound | Reported measurement | What showed up                                                |
| ------------------------ | -----------------------: | -------------------: | ------------------------------------------------------------- |
| 1-GPU decode             |                   16.8ms |               22.0ms | sustained HBM bandwidth, intermediate kernels, fixed overhead |
| 2-GPU decode             | 8.4ms plus communication |               13.5ms | smaller-shard efficiency and collectives                      |
| 1-GPU, 512-token prefill |        41.3ms dense math |               63.2ms | sub-peak matmul efficiency, attention, and cache writes       |

Usually in this order:

1. sustained HBM bandwidth is below the spec-sheet peak
2. intermediate and elementwise kernels are not free
3. kernel launches, sampling, and sync add fixed costs
4. real collectives have startup, topology, and bandwidth waste
5. matmul efficiency depends on exact dimensions and tiling

The formulas did not fail. They named which resource would saturate and which bottleneck would appear next. Arithmetic tells you which regime to look at. Profiling gives you the constants for one model, runtime, and machine.

---

## Checklist, for any model

Work in this order. Each step feeds the next.

**1. Weight capacity.**

$$M_{\text{weights}} = P \cdot b_{\text{weight}}$$

Ours: `52e9 × 2 = 104GB`, so at least 3 A100-40GBs, and 4 for clean head division.

**2. KV-cache cost per token, then per live token.**

$$
M_{\text{KV/token}}
= 2b_{\text{cache}}n_{\text{layers}}n_{\text{kv-heads}}d_{\text{head}}
$$

Ours: 2MiB/token, so 32GiB for 16,384 live tokens. Multiply by the sum of all live prompt **and** generated tokens, never by the request count.

**3. The dense roofline at your achievable batch.**

$$
T_{\text{large matrices}}
\approx
\max\left(
\frac{M_{\text{weights}}}{N\cdot BW_{\text{HBM}}},
\frac{F_{\text{dense/token}}\cdot B}{N\cdot R_{\text{FLOP}}}
\right)
$$

Ours: `max(17.3ms, 0.66ms)` at `B=8`. Compare `B` against `R_FLOP / BW_HBM` (208 here) to see which side of the crossover you are on.

**4. Context-dependent attention, in both FLOPs and bytes.**

$$F_{\text{attention/decode}} \approx 4(t+1)d\,n_{\text{layers}}$$

Ours: 4.30 GFLOPs (4% of arithmetic) but 8GiB read per GPU (25% of step time). Long contexts can turn a weight-bound decode into a cache-bandwidth-bound decode.

**5. Tensor-parallel communication.** Estimate the payload `2Bd`, the collective count `2L`, ring volume `2(N-1)/N`, and startup `α` per collective. Then check the link roofline `d/(2(N-1))` against `R_FLOP / BW_link` to see whether more GPUs will still help.

**6. Measure the real stack.** Prefill latency, time to first token, inter-token latency, throughput, HBM use, and tail latency under the request-length mix you actually expect. Then compare against steps 3 to 5 to see which of the five gaps you are paying.

---

## Recap {#recap}

1. **Tensor parallelism fixes capacity and weight bandwidth, and adds collectives.** Four GPUs cut the weight floor from 69.3ms to 17.3ms, at the cost of two logical all-reduces per block, 128 per step. Small batches pay startup (~1.0ms). Large batches also pay volume (~5.24ms at `B=500`).

2. **Our eight-conversation step sits around 23 to 24ms before software.** 17.3ms of weights plus 5.7ms of cache reads, with ~1.0ms of communication if it is fully exposed. That is about 348 aggregate tokens/s, or 41 to 43 per conversation. GQA would drop the cache-read term to 0.7ms.

3. **A bigger batch raises throughput and can hurt per-request latency.** `B=208` is the large-matrix sweet spot on this hardware. `B=500` vs 500 separate `B=1` steps is about a 192× gain in aggregate work. It is still 500 requests each producing one token.

4. **A link roofline says when more GPUs stop helping.** The A100's compute-to-link ratio is ~1040 FLOPs/byte. Attention's output projection gets `d/(2(N-1))`: 1,365 at `N=4`, 585 at `N=8`, 273 at `N=16`. Per-GPU work shrinks. The activation exchange does not.

5. **Dense FLOPs per token ≈ 2 × dense parameters.** Six matrices per block give `24d²`, so 64 blocks cost 103.08 GFLOPs/token against 51.54B parameters. Left out: cached attention (4.30 GFLOPs at 2,048 tokens), the vocabulary head (0.823 GFLOPs), and the small kernels whose cost is bytes rather than FLOPs.

6. **Benchmarks are slower for five predictable reasons.** Sub-peak bandwidth, intermediate kernels, fixed overhead, real collectives, and shape-dependent matmul efficiency. A 13B run with a 16.8ms lower bound measured 22.0ms. Its two-GPU step fell only to 13.5ms, not half.

For another model or machine:

1. Do the weights fit?
2. How many live tokens fit after the weights?
3. At that batch, is weight movement or arithmetic slower?
4. What communication, cache, and software costs stay exposed?
5. What does the real benchmark say?

---

## References

- [Part 1: Memory and Batching](/blog/2026/transformer-inference-arithmetic-part-1/)
- [kipply, "Transformer Inference Arithmetic"](https://kipp.ly/p/transformer-inference-arithmetic)
- [kipply, "Transformer Parameter Counting"](https://kipp.ly/p/transformer-param-count)
- [NVIDIA A100 Tensor Core GPU architecture](https://www.nvidia.com/en-us/data-center/a100/)
- [NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198), tensor-parallel communication structure
- [Ivanov et al., "Data Movement Is All You Need"](https://arxiv.org/abs/2007.00072)
- [Dao et al., "FlashAttention"](https://arxiv.org/abs/2205.14135)
- [Counting Transformer Parameters](/blog/2026/counting-transformer-parameters/)
