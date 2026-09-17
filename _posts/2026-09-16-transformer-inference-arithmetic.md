---
layout: post
title: "Transformer Inference Arithmetic — A Worked Walkthrough"
subtitle: "How KV cache, batching, memory bandwidth, and multi-GPU serving fit together—using one 52B model and one running workload."
date: 2026-09-16
categories: [Tech, machine-learning, llm]
tags: [machine-learning, llm, transformers, inference, gpu]
reading_time: 26
description: "A beginner-first walkthrough of Transformer inference: what problem each mechanism solves, how the important costs connect, and what the numbers mean in one 52B example."
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

Every formula below answers one question in that chain. You do not need to memorize every number. On a first reading, follow the **question** and **carry-forward** notes; the derivations are there so the conclusions do not feel magical.

---

## Part 0 — The minimum vocabulary

These are the only terms needed to start:

- A **token** is one text piece understood by the model. It can be a word, part of a word, punctuation, or whitespace.
- A **vector** is a list of numbers. One token is represented inside this model by 8,192 numbers.
- An **embedding** is the learned lookup that turns a token ID into that first vector.
- A **parameter** or **weight** is one learned number in the model. Most weights are arranged in rectangular grids called **matrices**.
- A **FLOP** is one floating-point operation. One multiplication plus one addition counts as roughly two FLOPs.
- A **Transformer block** is one repeated processing unit. **Attention** lets token positions gather information from one another. The **MLP** (multilayer perceptron) widens each token vector, transforms it, and shrinks it back.
- **Prefill** processes the known prompt. **Decode** generates new tokens afterward, one step at a time.
- A **KV cache** remembers attention data from earlier token positions so decode does not rebuild it.
- A **batch** contains token positions processed together. During decode, batch size `B=8` usually means eight conversations each producing one current token.
- **Latency** is how long one request or step waits. **Throughput** is how many total tokens the server produces per second.
- **HBM** is the GPU's large attached memory. **Capacity** asks how many bytes fit; **bandwidth** asks how many bytes can move per second.
- **Tensor parallelism** splits one model across several GPUs. Each GPU computes a piece, and some pieces must then be combined.
- **BF16** is the two-byte number format used for weights and cache values in this example.

The recurring symbols are:

- `d = d_model`: width of one token vector;
- `L = n_layers`: number of Transformer blocks;
- `S`: prompt length;
- `t`: earlier tokens visible during one decode step;
- `B`: token positions processed together;
- `N`: GPUs sharing the model.

### Our model and workload

The main walkthrough uses one illustrative 52B shape. Part 8 briefly switches to a published 13B benchmark only to compare theory with a real measurement.

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

This is an intentionally simple teaching example, not a claim that it is the best production configuration. Peak compute and bandwidth rates are upper limits, so time estimates derived from them are optimistic lower bounds.

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

For a prompt containing `S` tokens, the model carries an `S × 8,192` grid of numbers through the blocks. Each block keeps the same outer shape. It adds two kinds of updates:

- **attention** gathers useful information from other token positions;
- the **MLP** transforms each position on its own.

**Normalization** rescales a vector's values into a stable range. A **residual addition** adds a block's update to the incoming vector instead of replacing it. Both help information flow through 64 blocks without changing the 8,192-wide shape.

<img src="/assets/img/transformer-param-count/01-architecture.svg" alt="Decoder-only Transformer: tokenization and embedding, repeated attention and MLP blocks, then a vocabulary head." style="width: 100%; max-width: 54rem; height: auto; display: block; margin: 1.5rem auto;" />

### 1.1 Why attention creates Q, K, and V

Each block makes three different projections of a token vector:

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

Here `X` contains the current token vectors. Each `W` is a different learned matrix. A **projection** simply means multiplying by one of those matrices to create a new set of vectors.

A useful mental model is:

- **query:** what is the current position looking for?
- **key:** what does an earlier position advertise about itself?
- **value:** what content can that earlier position contribute?

For the current token, attention works in three steps:

1. compare its query with every visible key, producing one relevance score per position;
2. turn those scores into weights that sum to one;
3. take a weighted sum of the corresponding value vectors.

The distinction matters:

> `q · k` produces a relevance score, not a value. The score decides how much of the separately computed `v` to use.

The model splits this work across 64 attention heads so different heads can learn different kinds of relationships. Their outputs join back into one 8,192-wide vector.

### 1.2 Prefill and decode

The same model has two noticeably different inference phases.

**Prefill** processes the known prompt. All `S` prompt positions are available, so the GPU can process many of them in parallel. This phase also creates K and V for every prompt position.

**Decode** begins after prefill:

1. use the current token to compute its new Q, K, and V;
2. let its query attend to the cached earlier K/V entries plus its own new K/V;
3. choose the following token;
4. save the current K/V and repeat.

One conversation cannot generate all its future tokens in parallel: token 101 depends on token 100. Our eight conversations _can_ advance together, however. Each contributes one current position, giving the server a decode batch of `B=8`.

> **Carry forward:** Earlier K and V vectors are needed again on every later decode step. Recomputing them would repeat known work, so the server needs somewhere to remember them.

---

## Part 2 — The KV cache, quantified

> **Question:** Why does serving memory grow as conversations get longer?

For a token at position `i`, its query is needed while computing position `i`'s attention output. A later position `t` creates its own query, so it has no use for `q_i`.

The same later position _does_ need `k_i` and `v_i`: it compares `q_t` with `k_i`, then uses the resulting weight to decide how much of `v_i` to gather. Every future position may repeat that read.

Causality makes storage safe. At a given layer, position `i` cannot see positions after `i`; therefore adding a future token cannot change the K or V already computed for position `i`. For that request and layer, those vectors are immutable.

<img src="/assets/img/transformer-inference-arithmetic/01-kv-cache.svg" alt="A token's query is used once and discarded, while its key and value persist in the cache and are reused by future queries." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

The KV cache is simply saved work: keep an intermediate result because later steps will request exactly the same result. It does not change the model's answer.

### 2.1 What one cached token costs

With the standard multi-head attention assumed in our model table, each token needs one K vector and one V vector in every block. In BF16:

$$
\begin{aligned}
\text{KV bytes/token}
&= 2_{\text{K,V}}
   \cdot 2_{\text{bytes/value}}
   \cdot 64_{\text{blocks}}
   \cdot 64_{\text{heads}}
   \cdot 128_{\text{values/head}} \\
&= 2 \cdot 2 \cdot 64 \cdot 64 \cdot 128 \\
&= 2{,}097{,}152\text{ bytes} \\
&= 2\text{ MiB}.
\end{aligned}
$$

The factors have a direct meaning: two saved objects (`K` and `V`), two bytes per number, 64 blocks, and `64 × 128 = 8,192` values in each complete K or V vector.

Because this model has 64 KV heads whose widths add back to `d_model`, the shortcut is

$$\boxed{\text{KV bytes/token} = 4\,n_{\text{layers}}d_{\text{model}}.}$$

One 2,048-token conversation therefore needs

$$2\text{ MiB/token} \times 2048 = 4\text{ GiB}$$

of cache. Our eight active conversations need

$$8\times4\text{ GiB}=32\text{ GiB}.$$

Every decode step adds one cached token to each conversation, so the eight-request batch grows by another

$$8\times2\text{ MiB}=16\text{ MiB per step}.$$

This is why a memory manager counts **live tokens**, not merely requests. Eight short conversations and eight very long conversations have completely different cache costs.

### 2.2 What the cache saves—and what it does not

Without a cache, every new decode step would rebuild old intermediate state that has not changed. With the cache, the large projections and MLP work run only for the current token positions.

The cache does not make old context free. The new query must still:

- read earlier K/V entries;
- compare against the earlier keys;
- combine the earlier values.

Longer context therefore still increases attention work and cache traffic. The cache removes wasteful recomputation; it does not remove the cost of looking back.

> **Carry forward:** The cache solves a compute problem by creating a memory problem. We already need 32GiB for our eight conversations, and we have not yet counted the model's weights.

---

## Part 3 — Why decode often waits on memory, not math

> **Question:** An A100 can perform 312 trillion operations per second. Why can generating one token still take many milliseconds?

The phrase "load the model" hides two very different data movements.

**Event 1: storage → HBM.** At server startup, the checkpoint moves from disk or host memory into GPU HBM. This can take seconds, but it happens before requests are served, so it is not part of normal per-token latency.

**Event 2: HBM → compute units.** During every forward pass, the GPU fetches small weight tiles into its much smaller on-chip storage, uses them, and makes room for the next tiles.

<img src="/assets/img/transformer-inference-arithmetic/04-hardware-data-path.svg" alt="Model weights move from storage to GPU HBM once at startup, then weight tiles stream from HBM through small on-chip memory to compute units on every forward pass." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

_[Open the hardware data-path diagram at full size](/assets/img/transformer-inference-arithmetic/04-hardware-data-path.svg)._

Think of HBM as a warehouse beside a factory. Startup stocks the warehouse once. Every decode step still brings the needed pallets to the factory floor. The weights remain on the GPU, but they are not all beside the arithmetic units.

For 52B BF16 parameters:

$$52\text{e}9\text{ parameters}\times2\text{ bytes/parameter}=104\text{e}9\text{ bytes}=104\text{ GB}.$$

Reading that much data through one A100's 1.5TB/s HBM interface would take at least

$$T_{\text{weights,1}} = \frac{104\text{e}9}{1.5\text{e}12} = 69.3\text{ ms}.$$

The model does not actually fit on one 40GB A100. Across four GPUs, each GPU owns and reads roughly 26GB:

$$T_{\text{weights,4}} = \frac{26\text{e}9}{1.5\text{e}12} = 17.3\text{ ms}.$$

Now compare that with arithmetic. The model's six large matrices need about 103.08 billion FLOPs per token; Part 7 derives this number. Split across four ideal A100s:

$$T_{\text{math,token}}=\frac{103.08\text{e}9}{4\times312\text{e}12}\approx0.083\text{ ms}.$$

For our eight-conversation batch:

$$T_{\text{math},B=8}=8\times0.083\approx0.66\text{ ms}.$$

The GPUs need only 0.66ms of ideal arithmetic but about 17.3ms to stream their weight shards. **Weight movement is the bottleneck.** This is the problem batching tries to solve.

### 3.1 Why batching amortizes weight traffic

If `B` positions use a matrix together, the GPU reads the weights once for that operation and applies them to all `B` positions. The weight traffic stays roughly fixed while useful arithmetic grows with `B`. Let `P` mean the number of weights in that matrix:

$$
\text{arithmetic intensity} \approx
\frac{2BP\text{ FLOPs}}{2P\text{ bytes}}
= B\text{ FLOPs/byte}.
$$

**Arithmetic intensity** means useful operations performed per byte moved.

In plain language, a batch of 8 gets about eight uses from each fetched weight; a batch of 208 gets about 208 uses. The A100's own compute-to-bandwidth balance is

$$\frac{312\text{e}12\text{ FLOP/s}}{1.5\text{e}12\text{ byte/s}} \approx 208\text{ FLOPs/byte}.$$

Matching the workload's `B` FLOPs/byte with the hardware's 208 FLOPs/byte gives the ideal **crossover** near `B=208`:

- **Below 208:** weights cannot arrive quickly enough to keep all arithmetic units busy. The operation is memory-bandwidth-bound.
- **Near 208:** weight delivery and arithmetic take similar time. Hardware utilization is best in this simplified model.
- **Above 208:** the arithmetic units are full, so adding more positions increases step time. The operation is compute-bound.

During decode, `B` means concurrent conversations contributing one current token each. It does **not** mean one conversation generates 208 future tokens simultaneously.

<img src="/assets/img/transformer-inference-arithmetic/02-batching-crossover.svg" alt="Ideal batching roofline for the 52B model on four A100 GPUs: a 17.3 millisecond weight-streaming floor meets the compute line near a batch of 208." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

_[Open the batching crossover graph at full size](/assets/img/transformer-inference-arithmetic/02-batching-crossover.svg)._

The graph uses `TP=4` as shorthand for four-way tensor parallelism.

Using only the six large matrices:

| Positions processed together (`B`) | Weight floor, 4 GPUs | Ideal math time, 4 GPUs | Large-matrix lower bound |
| ---------------------------------: | -------------------: | ----------------------: | -----------------------: |
|                                  1 |              17.3 ms |                0.083 ms |                  17.3 ms |
|                       **8 (ours)** |          **17.3 ms** |             **0.66 ms** |              **17.3 ms** |
|                            **208** |          **17.3 ms** |             **17.2 ms** |             **≈17.3 ms** |
|                                500 |              17.3 ms |                 41.3 ms |                  41.3 ms |

The `B=500` row only shows what happens beyond the crossover. Four A100s could not hold 500 of our 2,048-token conversations; such a batch would require much shorter contexts or substantially more memory.

The lower bound uses

$$T_{\text{large matrices}}\approx\max(T_{\text{weight read}},T_{\text{math}}),$$

because optimized GPU programs overlap weight fetching with arithmetic. Positions below 208 are not literally free; they use compute capacity that would otherwise sit idle while weights arrive. Cache reads, attention, communication, and software still add cost.

> **Carry forward:** Our `B=8` workload is far below the ideal 208-position crossover. More concurrent conversations could improve throughput—but every additional conversation needs KV-cache memory.

---

## Part 4 — Capacity: does it fit?

> **Question:** We want a larger batch, but can the weights and all live caches fit in HBM together?

The weights alone need at least

$$\left\lceil\frac{104}{40}\right\rceil = 3\text{ A100-40GB GPUs}.$$

But "the weights fit" is not enough. Whatever remains must hold the KV cache and the runtime's temporary buffers:

| GPUs | Total HBM | Weight memory | Theoretical remainder | Theoretical KV-token capacity |
| ---: | --------: | ------------: | --------------------: | ----------------------------: |
|    3 |    120 GB |        104 GB |                 16 GB |                 ≈7,629 tokens |
|    4 |    160 GB |        104 GB |                 56 GB |                ≈26,703 tokens |

Our eight conversations already contain

$$8\times2{,}048=16{,}384\text{ live tokens}.$$

So three GPUs can hold the weights but **cannot hold our workload's cache**. Four GPUs can: the cache uses 32GiB (about 34.4GB in decimal units), leaving roughly 21.6GB before runtime overhead.

The 26,703-token ceiling also reveals something important about batching. If every conversation is 2,048 tokens long, four GPUs can theoretically retain only

$$\left\lfloor\frac{26{,}703}{2{,}048}\right\rfloor=13\text{ such conversations}.$$

That would give a decode batch around 13—still nowhere near the ideal 208-position crossover from Part 3. Long contexts consume the memory that would otherwise support more concurrent requests.

And 13 is only a mathematical ceiling. Real serving also needs memory for temporary activations (intermediate token vectors), communication buffers, the GPU runtime, and unused safety space. The safe limit is lower.

This leads to the practical tradeoff:

- longer conversations consume more cache per request;
- fewer active requests mean smaller decode batches;
- smaller batches reuse each weight load less efficiently.

> **Carry forward:** Four GPUs solve our capacity problem. But no single GPU now computes the full answer, so the devices must exchange partial results.

---

## Part 5 — Splitting weights across GPUs

> **Question:** If four GPUs each compute only part of a matrix multiplication, how does the model recover one correct answer?

Tensor parallelism shards each large matrix. With four GPUs, each package owns roughly one quarter of the weight bytes and performs roughly one quarter of the matrix-multiplication work:

$$104\text{ GB}/4=26\text{ GB of weights per GPU}.$$

Because every GPU has its own HBM channels and compute units, this is more than pooling capacity: four devices can stream four shards and multiply them in parallel.

<img src="/assets/img/transformer-inference-arithmetic/03-tensor-parallel.svg" alt="Four tensor-parallel GPUs each read a 26GB weight shard, compute a partial output, and exchange partials in a collective to reconstruct the full activation." style="width: 100%; height: auto; display: block; margin: 1.5rem auto;" />

Imagine splitting a long arithmetic sum among four people. Each person can calculate one quarter independently, but nobody has the final total until the four partial sums are combined. Sharded matrix multiplication has the same dependency.

In one common sharding arrangement, a Transformer block needs two logical combine operations:

1. after attention's output projection;
2. after the MLP's down-projection.

The operation that exchanges and sums the partial answers is called an **all-reduce**. With 64 blocks, one decode step reaches

$$2\times64=128\text{ logical all-reduces}.$$

That sounds expensive, but communication has two separate costs.

### 5.1 Small messages pay startup cost

Starting a collective has a fixed latency even when its payload is small—like establishing a phone call before speaking.

For our `B=8` workload, one **activation payload**—the current `B × d` grid of intermediate token vectors—is only

$$2\text{ bytes}\times8\times8192=131{,}072\text{ bytes}=128\text{ KiB}.$$

If one collective has an optimistic `8μs` startup cost, 128 calls contribute roughly

$$128\times8\mu\text{s}\approx1.0\text{ ms}.$$

At this small batch, fixed startup matters more than the tiny amount of data.

### 5.2 Large messages also pay for bytes moved

For the same hypothetical `B=500` point shown on the graph—possible only with shorter contexts or more memory—one activation payload grows to

$$2\times500\times8192=8.192\text{ MB}.$$

A simple four-GPU ring estimate gives about **5.24ms** of communication-by-volume across all 128 collectives, before their startup cost. The exact value depends on links, topology, and the collective implementation; the important idea is the shape of the cost:

- small batch: mostly "start 128 exchanges";
- large batch: startup **plus** moving much larger activations.

<details markdown="1">
<summary>Optional: where the 5.24ms estimate comes from</summary>

For a ring all-reduce over `N` GPUs, bytes sent per GPU are approximately

$$M_{\text{ring}}\approx2\frac{N-1}{N}M_{\text{payload}}.$$

At `N=4`, one 8.192MB payload causes about 12.288MB to be sent per GPU. Across 128 collectives:

$$12.288\text{ MB}\times128\approx1.57\text{ GB}.$$

At 300GB/s:

$$1.57\text{ GB}/300\text{ GB/s}\approx5.24\text{ ms}.$$

</details>

### 5.3 Why more GPUs do not keep halving latency

Moving from four to eight GPUs halves each weight shard from 26GB to 13GB, so the ideal weight-read floor falls from 17.3ms to about 8.7ms. But the 8,192-wide activation still has to be combined, and now more GPUs participate.

More GPUs reduce local memory traffic and arithmetic. They do not remove collective startup or make the exchanged activation disappear. Eventually communication becomes large relative to each GPU's shrinking amount of local work.

> **Carry forward:** A decode step is not "compute time plus every other number." Weight reads and arithmetic overlap, while some communication is exposed between dependent stages. We need one latency model that keeps those relationships straight.

---

## Part 6 — Putting step latency together

> **Question:** Which costs overlap, which costs add delay, and what do they mean for one user versus the whole server?

Start with the six large matrices. Their ideal time is the slower of weight delivery and arithmetic:

$$
T_{\text{large matrices}}
\approx
\max(T_{\text{weight read}},T_{\text{math}}).
$$

We take the maximum—not the sum—because an optimized matrix-multiplication program computes on one tile while fetching another. The slower stream determines the floor.

A complete step then looks like

$$
T_{\text{step}}
\approx T_{\text{large matrices}}
+ T_{\text{exposed communication}}
+ T_{\text{attention and cache}}
+ T_{\text{software overhead}}.
$$

Only communication that cannot hide behind useful work belongs in the exposed term. This is why neither "add every time" nor "take one maximum for the entire model" is always correct.

### 6.1 Our eight-conversation step

We already calculated:

- weight-read floor: **17.3ms**;
- ideal math for `B=8`: **0.66ms**.

Therefore the large-matrix part is about 17.3ms. Now include the long context: across all eight conversations, standard attention reads roughly the 32GiB K/V cache during one decode step. Sharded evenly across four GPUs, that is 8GiB per GPU. At an ideal 1.5TB/s:

$$T_{\text{KV read}}\approx\frac{8\text{ GiB}}{1.5\text{ TB/s}}\approx5.7\text{ ms}.$$

The weights and cache share the same HBM bandwidth, so together they create a memory-traffic floor near

$$17.3+5.7=23.0\text{ ms}.$$

Communication may partly overlap. If the toy 1.1ms communication cost is fully exposed, the estimate becomes 24.1ms before software overhead. Implementations can change exact traffic and overlap, but this calculation proves that long-context cache reads are not a rounding error.

The step produces eight next-token positions, so the 23.0ms memory floor gives an aggregate throughput ceiling near

$$8/0.0230\approx348\text{ tokens/s}.$$

With fully exposed communication, that falls to about 331 aggregate tokens/s, or 41 tokens/s per continuously active conversation. Real throughput will be lower after software overhead.

### 6.2 Why a larger batch helps throughput but can hurt latency

Suppose the requests had much shorter contexts—or the server had more memory—so it could form `B=208`. The large-matrix weight and math terms would both be about 17.3ms, giving a large-matrix-only ceiling near 12,000 positions/s before cache, communication, and software costs.

That is far more aggregate work than our `B=8` case, but a request may wait while the scheduler forms the batch. The tradeoff is:

- **larger batch:** better total hardware efficiency;
- **individual request:** potentially more queueing and a longer step.

Batching improves throughput only when enough live work exists and the cache for that work fits. It never lets one conversation skip the sequential dependency between its own output tokens.

> **Carry forward:** The 103.08-GFLOP figure has powered every math-time estimate so far. Next we derive it from the model's matrices so it is not just a magic number.

---

## Part 7 — Where "FLOPs per token ≈ 2 × parameters" comes from

> **Question:** Why did we use 103.08 billion FLOPs per token in Parts 3 and 6?

### 7.1 One small matrix explains the rule

Multiplying an `m × n` matrix by a length-`n` vector performs approximately

$$2mn\text{ FLOPs},$$

counting one multiply and one add per matrix element. The matrix itself contains `mn` parameters. Therefore, if a weight is used once in a matrix-vector multiplication,

$$\text{FLOPs} \approx 2 \times \text{parameters}.$$

For example, a `2 × 3` matrix contains six parameters. A literal matrix-vector multiply uses six multiplications and four additions. The conventional `2mn` estimate rounds this to 12 FLOPs by treating each weight as one multiply-accumulate pair. The two-operation difference disappears at the thousands-wide dimensions used here.

### 7.2 Apply the same rule to one Transformer block

We can group the six large matrices instead of memorizing them individually:

| Part of one block | Large matrices                             | Parameters | FLOPs per token |
| ----------------- | ------------------------------------------ | ---------: | --------------: |
| Attention         | Q, K, V, and output: four `d × d` matrices |      `4d²` |           `8d²` |
| MLP               | one `d × 4d` and one `4d × d` matrix       |      `8d²` |          `16d²` |
| **Total**         | six matrices                               | **`12d²`** |      **`24d²`** |

For `d=8192`, one block costs

$$
24\times8192^2
\approx1.61\text{ billion FLOPs/token}.
$$

Across 64 blocks:

$$
24\times64\times8192^2
\approx103.08\text{ billion FLOPs/token}.
$$

Those block matrices contain half as many parameters:

$$103.08\text{B}/2=51.54\text{B parameters},$$

which is almost the entire 52B model. That is where the useful rule comes from:

$$\boxed{\text{dense FLOPs per token}\approx2\times\text{dense parameters}.}$$

### 7.3 Why `2P` is a baseline, not the whole step

Here `P` means parameter count, and **dense** means the ordinary full weight matrices counted above.

The shortcut counts the six dominant matrices. It leaves out:

- attention over the cached context;
- the final projection to vocabulary scores;
- normalization, activation, sampling, and data movement.

At a 2,048-token context, cached attention adds about 4.30 billion FLOPs, roughly 4.2% of the 103.08-billion dense baseline. The vocabulary projection adds about 0.82 billion more. Both are real, but the six large block matrices still explain most arithmetic in this example.

> **Carry forward:** We can now trace the earlier 0.083ms math estimate directly back to six matrix groups. The remaining gap between our formulas and a real measurement comes from hardware never behaving like a perfect specification sheet.

---

## Part 8 — Why real hardware is slower than the clean formulas

> **Question:** If the arithmetic is correct, why do benchmarks report larger times?

Our equations deliberately calculate lower bounds. Real runs lose time in a few predictable places:

1. **Sustained bandwidth is below the specification.** A real GPU operation rarely gets 100% of the advertised 1.5TB/s.
2. **Small operations still move data.** Normalization, residual additions, and activations may write a tensor only for the next operation to read it back.
3. **Long context means cache traffic.** Our 2,048-token request owns 4GiB of K/V state across all blocks, and attention reads the relevant parts during decode.
4. **GPU operations and collectives have setup cost.** Launches and synchronization take time even when their arithmetic is tiny.
5. **Matrix shape affects utilization.** A GPU does not reach 312TFLOP/s for every matrix dimension and batch size.

Serving runtimes therefore combine small operations when possible, keeping intermediate data on-chip and reducing HBM round trips even when the mathematical operation count stays the same.

### 8.1 A real benchmark

kipply reported a FasterTransformer benchmark for a 13B-shaped model with width 5,120, 40 blocks, and a 512-token context. The same method predicts 25.17 billion dense FLOPs per token.

| Workload                 |       Simple lower bound | Reported measurement | Why reality was slower                                      |
| ------------------------ | -----------------------: | -------------------: | ----------------------------------------------------------- |
| 1-GPU decode             |                   16.8ms |               22.0ms | lower sustained bandwidth, small operations, fixed overhead |
| 2-GPU decode             | 8.4ms plus communication |               13.5ms | collectives and less efficient smaller shards               |
| 1-GPU, 512-token prefill |                   41.3ms |               63.2ms | sub-peak matrix operations, attention, and cache writes     |

The two-GPU result is especially useful. Halving each weight shard reduced the ideal weight time from 16.8ms to 8.4ms, but the measured time fell only from 22.0ms to 13.5ms. Communication and costs that were not divided across GPUs became a larger fraction of the step.

The lesson is not that the formulas failed. They correctly identified what should improve and which bottleneck would appear next. Their job is to classify the regime; profiling supplies the real constants.

> **Carry forward:** We now have every piece. The final summary reconnects them in the order one request experiences them.

---

## TL;DR — The whole story in one pass {#tldr}

1. **A request begins with prefill and continues with decode.** Prefill reads the known prompt and creates attention state. Decode then generates one token per active conversation per step; future tokens from one conversation cannot be generated simultaneously.

2. **The KV cache exists because K and V are reused.** A token's query is used for its own attention result, while later tokens repeatedly need its key and value. Saving K/V avoids rebuilding unchanged work. In this model the cache costs **2MiB per live token**.

3. **Our workload already needs 32GiB of cache.** Each 2,048-token conversation needs 4GiB; eight conversations need 32GiB. Every eight-request decode step adds another 16MiB.

4. **The 52B BF16 weights need 104GB.** One 40GB A100 cannot hold them. Four A100s give each GPU a 26GB shard; after our 32GiB cache, about 21.6GB of aggregate HBM remains before runtime overhead.

5. **Small-batch large-matrix computation waits mostly on weight movement.** On four A100s, reading the weight shards has an ideal 17.3ms floor while arithmetic for `B=8` needs only 0.66ms. Our long contexts add about 5.7ms of cache-read traffic, creating a 23.0ms memory floor; fully exposed toy communication raises the pre-software estimate to about 24.1ms.

6. **Batching reuses one weight stream across more positions.** The ideal A100 crossover is near `B=208`. Below that point, a larger batch mainly fills otherwise-idle arithmetic capacity; above it, arithmetic starts increasing step time. `B` means concurrent current positions, not future tokens from one response.

7. **Cache capacity limits how large a decode batch can become.** Four GPUs can theoretically retain about 26,700 of these cached tokens. At 2,048 tokens per conversation, that is only about 13 conversations before runtime overhead—far below the 208-position arithmetic sweet spot. Longer context can therefore reduce throughput by reducing concurrency.

8. **Tensor parallelism solves capacity and weight bandwidth but adds communication.** Four GPUs cut the ideal weight-read floor from a one-device equivalent of 69.3ms to 17.3ms. They also perform 128 logical combine operations per step in this model. Small batches mostly pay collective startup; large batches also pay to move larger activations.

9. **Large-matrix FLOPs per token are about twice the large-matrix parameter count.** Every matrix weight contributes roughly one multiply and one add when used. The six large matrices across 64 blocks contain about 51.54B parameters and require about 103.08B FLOPs per token.

10. **The formulas identify the bottleneck; benchmarks give the real latency.** Actual hardware sustains less than peak bandwidth and compute, while cache traffic, small operations, communication, and software add time. A 13B run with a 16.8ms lower-bound estimate measured 22.0ms—different in magnitude, but for understandable reasons.

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
