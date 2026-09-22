---
layout: post
title: "Prefill vs Decode"
subtitle: "Why prompts are cheap, generation is expensive, and most of LLM serving is a response to that one asymmetry."
date: 2026-09-21
categories: [Tech, machine-learning, llm]
tags: [machine-learning, llm, systems, inference, kv-cache]
reading_time: 19
description: "An LLM request splits into prefill and decode. Prefill shares the weight-stream cost across the prompt, so it is compute-bound. Decode pays it for about one token, so it is memory-bandwidth-bound."
featured: true
---

> **One-paragraph version.** An LLM generates text one token at a time. Because attention is causal, an old token's K and V vectors never change, so we cache them (the **KV cache**). That splits every request into **prefill** (the whole prompt in one parallel pass; fills the cache; gives the first token) and **decode** (one token per pass). Every pass must stream _all_ the model weights from memory. Prefill shares that cost across N tokens, so it is **compute-bound**. Decode shares it across about 1 token, so it is **memory-bandwidth-bound**. Most of LLM serving (batching, KV memory management, chunked prefill) is a response to this one asymmetry.

[Jump to the key takeaways.](#key-takeaways-read-this-if-nothing-else)

---

## Key takeaways (read this if nothing else)

### The story in 8 steps

1. A model call is stateless: it only sees the tokens you pass in. Generating text means calling it, appending the sampled token, and calling it again.
2. Done naively, every step reprocesses the entire history. That is hugely wasteful.
3. Causal masking means position _j_ only depends on tokens 0..j, so appending tokens never changes anything already computed.
4. Future tokens need the **K and V** of past tokens (not Q). So we cache K and V: the KV cache.
5. At the start we have a whole known prompt and an empty cache, so we process all prompt tokens together. That is **prefill**. It outputs the cache and the first token.
6. After that, each step handles one new token that depends on the previous step, so it cannot be parallelized over time. That is **decode**.
7. Hardware view: every pass streams all weights (14 GB for a 7B fp16 model) from memory. Work per byte fetched equals the number of tokens sharing that fetch. Prefill has hundreds, so the chip is busy. Decode has about 1, so the chip mostly waits on memory.
8. The fix for decode is **batching** many requests into one pass. The catch is that each request has its own KV cache, so cache reads are not shared.

### The "why" questions, answered

**Why is prefill compute-bound?**
Every weight streamed from memory gets reused by all N prompt tokens before the next weight is needed. So the chip does about N operations per byte fetched (N = 512 gives 512, versus a ridge of 156). Memory delivers data faster than the math units can consume it, so the math is the bottleneck: time ≈ FLOPs / F (23 ms in the A100 example). This only holds when N is above the ridge. A very short prompt is memory-bound too.

**Why is decode memory-bound?**
Each step has only one new token per request, so every fetched weight is used once: about 1 operation per byte, versus a ridge of 156. The chip could do about 156x more math with each byte but has nothing to do, so it waits on memory: time ≈ bytes / BW (7 ms versus 0.045 ms of actual math, so about 99% idle). It is structural: you cannot add more tokens from the same request, because each depends on the previous one. The KV cache reads are memory-bound as well, at about 1 FLOP per byte.

**Why is prefill faster than decode (per token)?**
Both phases stream the same 14 GB of weights per pass. Prefill splits that cost across N tokens and decode pays it for 1 token. Per token, decode costs `2P / BW` and prefill costs `2P / F`, so the ratio is `F / BW`, the ridge point (about 156x on an A100). Concretely: 512 tokens take 23 ms via prefill and 3.6 s via decode. (One prefill _pass_ is slower than one decode _step_, 23 ms versus 7 ms, but it does 512 tokens of work.)

**What can we do about each?**

| Phase                   | Problem                    | Levers                                                                                                                                                              |
| ----------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Decode (memory-bound)   | Few tokens per weight read | Batch more requests together (continuous batching). Move fewer bytes (quantization, GQA / smaller KV cache). Speculative decoding (verify k tokens per weight read) |
| Prefill (compute-bound) | Big burst of math          | Only more compute makes the pass itself faster. Avoid redoing it (prefix caching). Stop it from stalling running decodes (chunked prefill, separate hardware pools) |

### Symbols used

| Symbol       | Meaning                                                                       |
| ------------ | ----------------------------------------------------------------------------- |
| `P`          | number of model parameters (7B model: 7×10⁹)                                  |
| `b`          | bytes per parameter (fp16 = 2)                                                |
| `N`          | prompt length in tokens                                                       |
| `B`          | number of requests decoded together (batch size)                              |
| `ctx`        | tokens currently in one request's KV cache                                    |
| `F`          | chip's peak compute, FLOP/s (A100 fp16: 312×10¹²)                             |
| `BW`         | chip's memory bandwidth, bytes/s (A100: about 2×10¹²)                         |
| `R = F / BW` | **ridge point**: work per byte needed to keep the chip busy (A100: about 156) |

### Formulas worth remembering

```
1. Weight bytes  = P × b                    (streamed from memory on EVERY pass)
2. FLOPs/token   ≈ 2 × P                    (one multiply + one add per parameter)
3. Pass time     ≈ max( bytes / BW ,  FLOPs / F )   (slowest resource wins)
4. FLOPs per byte ≈ tokens sharing a weight read      (fp16)
                    → compare with R:  below R = memory-bound, above R = compute-bound
5. Decode step   ≈ ( weights + B × ctx × KV-per-token ) / BW
6. KV per token  = 2 × layers × kv_heads × head_dim × b
```

In words: 3 says time is set by whichever of "moving bytes" or "doing math" is slower. 4 says which one that is depends on how many tokens share each weight read. 5 says decode time is a fixed weight cost plus a KV cost that grows with batch and context.

### Ideas worth remembering

- Prefill means many tokens per weight read. Decode means one.
- Decode speed is set by bytes moved, not by math.
- A generated token costs about R times more than a prompt token (about 156x on an A100). Prompts are cheap, generation is expensive.
- Batching decode is nearly free while weight traffic dominates. It stops being free when KV traffic catches up.
- Prompts shorter than about R tokens are memory-bound too, and cost about one decode step.
- The KV cache grows linearly with length, per request. It is what limits how many requests fit in memory.
- User-facing: **TTFT** (time to first token) is queue wait plus prefill. **TPOT** (time per output token) is one decode step.

### Numbers to remember (7B fp16 on an A100, ideal)

| Quantity                   | Value                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| Weights                    | 14 GB                                                                                     |
| One decode step, batch 1   | about 7 ms (memory time). The math needs only 0.045 ms, so about 99% of compute sits idle |
| One request's decode speed | about 143 tokens/s                                                                        |
| Prefill of 512 tokens      | about 23 ms, roughly 3 decode steps' worth                                                |
| Same 512 tokens via decode | about 3.6 s, about 156x slower                                                            |
| KV cache                   | 0.5 MB per token, so about 2 GB for 4096 tokens                                           |

> If someone asks "what's the difference between prefill and decode?"
>
> "An LLM generates text one token at a time, and it caches the keys and values of past tokens so it doesn't recompute them. Prefill is the first phase: it processes the whole prompt in one big parallel pass, fills that cache, and produces the first token. Decode is everything after that: one token per pass, reading the cache and the weights each time. Every pass has to stream all the model weights from memory, so in prefill that cost is shared across hundreds of tokens and the GPU is compute-bound, while in decode it's spent on one token, so the GPU mostly waits on memory. That's why prompts are cheap, generation is slow, and why serving systems batch many requests together."

---

## 1. Generation is a loop around a stateless function

A transformer call takes a token sequence. Each position gets a vector that is refined layer by layer. The last position's vector goes through the LM head to give scores over the vocabulary, which is the next-token prediction. Attention is causal: position _i_ only sees positions ≤ _i_.

The model has **no memory between calls**. It is a pure function of the tokens you give it. So generation is a loop:

> predict next token → append it → predict again → …

## 2. What the naive loop wastes

Prompt `The cat sat`:

| Call | Tokens fed in          | Output |
| ---- | ---------------------- | ------ |
| 1    | The cat sat (3)        | on     |
| 2    | The cat sat on (4)     | the    |
| 3    | The cat sat on the (5) | mat    |

Every call reprocesses all earlier tokens from scratch. With a 1000-token prompt and 200 generated tokens, the last call pushes about 1200 tokens through every layer, and about 1199 of them were already processed in the previous call.

So the real question is: when we append a token, does anything about the old tokens actually change?

## 3. The past never changes, so we cache K and V

Trace an old token _j_ through the model. Its layer-1 input depends only on itself. Its attention output depends only on tokens 0..j, because of the causal mask. That output is the next layer's input, so the same argument holds at every layer.

<img src="/assets/img/prefill-vs-decode/01-kv-cache-causality.svg" alt="Causal masking means position j only sees tokens 0 through j, so its K and V never change and can be cached." style="width: 100%; max-width: 28rem; height: auto; display: block; margin: 1.5rem auto;" />

**Appending tokens at the end can never change what happened at earlier positions.** (This is why the trick fails for bidirectional models like BERT.)

What does a new token need from the past? It compares its own query against all earlier **keys** and takes a weighted sum of their **values**. It never needs old queries: a query means "what am I looking for", and only the token doing the looking uses it. Keys and values are needed by every future token. So we keep K and V for every token at every layer: the **KV cache**.

Size per token: `2 × layers × kv_heads × head_dim × bytes`. For a 7B model that is 0.5 MB per token, so a 4096-token request holds about 2 GB of cache.

## 4. Two jobs: prefill and decode

<img src="/assets/img/prefill-vs-decode/02-prefill-decode-flow.svg" alt="Prefill processes the whole prompt, fills the KV cache, and emits the first token. Decode then takes one token per step, reading and appending to the cache." style="width: 100%; max-width: 54rem; height: auto; display: block; margin: 1.5rem auto;" />

**Prefill (N prompt tokens):**

1. Embed to an N×d matrix.
2. In each layer, one big matmul produces Q, K, V for all N tokens. Write K and V into the cache.
3. Compute scores Q·Kᵀ (N×N), apply the causal mask, softmax, multiply by V.
4. Output projection, residual and FFN (more big matmuls), then the next layer.
5. After the last layer, take only the final position, apply the LM head, sample. That is token 1.

**Decode (one step):**

1. Embed the single new token to a 1×d vector.
2. In each layer, compute its q, k, v. Append k and v to the cache.
3. Compute scores q·Kᵀ against all cached keys (a 1×(ctx+1) row), softmax, weighted sum of cached values.
4. Output projection and FFN on that one vector, then the next layer.
5. LM head, sample, feed the token into the next step.

**Why prefill can be parallel:** all inputs are known and causal masking makes each position independent of later ones, so computing them all at once gives the same result as one at a time. It is the same computation as a training forward pass. **Why decode cannot:** token t+1's input is the output of step t.

## 5. The hardware picture: two speeds, one bottleneck

Compute units can only work on data that is on-chip, and on-chip storage is tens of MB. A 7B fp16 model is 14 GB, so it cannot stay there. On **every** forward pass the weights are streamed from main memory (HBM on an A100, unified memory on a Mac) through the compute units.

A chip therefore has two independent speeds:

- **Compute rate** `F`: A100 about 312 trillion FLOP/s.
- **Memory bandwidth** `BW`: A100 about 2 trillion bytes/s.

They work in parallel like a chef and a delivery truck, so a pass takes roughly whichever is slower. Their ratio `R = F / BW = 312e12 / 2e12 ≈ 156`. In the time it takes to fetch one byte, the chip could do about 156 operations on it. To keep the chef busy, each delivered byte must be used about 156 times.

## 6. Why "tokens per weight read" is the key quantity

Take one weight matrix W (d×d, fp16) and a matrix X with n rows (one row per token in this pass):

```
bytes to move = d² × 2                      (each weight is 2 bytes)
FLOPs to do   = n × d² × 2                  (each row: d² multiply-adds, 2 FLOPs each)
FLOPs per byte = (2·n·d²) / (2·d²) = n
```

**Every weight you fetch gets used once per token in the pass.** So:

<img src="/assets/img/prefill-vs-decode/03-ridge-point.svg" alt="If n tokens share each weight read, compare n with the ridge R = F / BW. Below R the pass is memory-bound; above R it is compute-bound." style="width: 100%; max-width: 40rem; height: auto; display: block; margin: 1.5rem auto;" />

- **Prefill:** n = N (say 512), well above 156, so the chef is busy: compute-bound.
- **Decode, one request:** n = 1. The chip could do 156x more math with each fetched byte, but there is nothing more to do. The step lasts exactly as long as the streaming: memory-bound.

## 7. Worked example: 7B fp16 on an A100

### Inputs

```
P  = 7×10⁹ parameters, 2 bytes each (fp16)
F  = 312×10¹² FLOP/s
BW = 2×10¹² bytes/s
```

### Two constants

```
Weight bytes   = 7×10⁹ × 2 = 14×10⁹ bytes = 14 GB
FLOPs per token ≈ 2 × P    = 14×10⁹  = 14 GFLOPs
```

Why 2P: a linear layer with d_in × d_out weights does d_in multiplies and about d_in adds for each of d_out outputs, so about 2 FLOPs per weight per token. (This ignores attention over the cache, which is small here.)

### One decode step (1 token)

```
memory time  = 14×10⁹ / 2×10¹²      = 7 ms
compute time = 14×10⁹ / 312×10¹²    ≈ 0.045 ms
step time    = max(7, 0.045)        = 7 ms
compute busy = 0.045 / 7            ≈ 0.64%   → about 99% idle
one user's speed = 1 / 0.007 s      ≈ 143 tokens/s
```

### Prefill of 512 tokens

```
FLOPs        = 512 × 14×10⁹         ≈ 7.2×10¹²
compute time = 7.2×10¹² / 312×10¹²  ≈ 23 ms
memory time  = 14 GB / 2 TB/s       = 7 ms   (weights streamed once for all 512 tokens)
pass time    = max(23, 7)           = 23 ms  → compute-bound
```

The 7 ms of streaming overlaps with the math, so it hides underneath the 23 ms. Cross-check: FLOPs per byte = 7.2×10¹² / 14×10⁹ = 512, which is above R = 156. And 512 / 156 = 3.28, so the pass takes 3.28 × its 7 ms memory floor ≈ 23 ms. Both methods agree.

### The comparison

```
512 tokens via decode:  512 × 7 ms = 3584 ms ≈ 3.6 s
512 tokens via prefill: 23 ms
ratio ≈ 156
```

That ratio is the ridge point, and it is not a coincidence. Per token, decode costs `weight_bytes / BW` and compute-bound prefill costs `2P / F`. With fp16, `weight_bytes = 2P`, so:

```
(decode cost per token) / (prefill cost per token) = (2P/BW) / (2P/F) = F / BW = R
```

Consequences: prompts are cheap and generation is expensive. A prompt shorter than about 156 tokens is memory-bound too, so it costs about one decode step (7 ms) regardless of its length.

> **Ideal numbers.** These are ideal roofline figures. Real kernels reach roughly 50-70% of peak FLOPs and 70-90% of peak bandwidth, so real times are 1.2-2x worse. The ratios and shapes still hold.

## 8. Batching: the fix for decode, and its catch

Decode wastes the chip because each fetched weight serves 1 token. Tokens from the same request cannot be added (each depends on the previous one), but tokens from **other requests** can. Run B requests' next-token steps in one pass and each weight is fetched once and used B times. Work per byte goes from 1 to B at almost no extra time. That is why doubling the batch barely slows a decode step at small B.

**The catch:** each request has its own KV cache. Batching B requests reads B separate caches, and attention does only about 1 FLOP per byte of cache read, so it stays memory-bound at any batch size:

```
decode step ≈ ( weights + B × ctx × KV-per-token ) / BW
```

Example (7B, A100, ctx = 1024): one request's cache is about 0.54 GB. At batch 1 the step reads about 14.5 GB (about 7.3 ms). At batch 64 it reads about 48 GB (about 24 ms). That is about 19x the throughput, not 64x, because cache traffic (34 GB) now exceeds weight traffic (14 GB). The crossover is at about 26 requests.

Memory capacity is a separate limit: with 80 GB and 14 GB of weights, about 120 requests of 1024 tokens fit. This is why KV memory management (PagedAttention) matters.

## 9. Is prefill bound by anything?

Yes, normally **compute-bound**, with three caveats:

- It needs N above the ridge (about 150 tokens on an A100). Short prompts are memory-bound.
- Attention cost grows as N², so very long prompts cost more than proportionally.
- Batching prefills helps little: one big prefill already saturates the chip.

Two other limits to keep in mind: **memory capacity** (KV cache caps how many requests fit) and **overhead** (kernel launches and Python dominate for tiny models like GPT-2 small, so measured step times land far above the bandwidth prediction).

## 10. Prefill and decode interfere

<img src="/assets/img/prefill-vs-decode/04-prefill-decode-interfere.svg" alt="A long prefill stalls every in-flight decode. Chunked prefill slices that work across decode steps so TPOT stays steady." style="width: 100%; max-width: 54rem; height: auto; display: block; margin: 1.5rem auto;" />

- **Chunked prefill:** split a long prompt into slices and add one per decode step. TPOT stays steady; the new request's TTFT gets slightly worse.
- **Prefill/decode disaggregation:** run the two phases on separate hardware pools, since one is compute-bound and the other bandwidth-bound.
- **Speculative decoding:** a small draft model proposes k tokens and the big model verifies all k in one pass. That looks like a mini-prefill, so it uses the idle compute in decode.
- **Prefix caching:** skip prefill for a prompt prefix that is already cached.

---

## Further reading

- [Kipply, "Transformer Inference Arithmetic"](https://kipp.ly/p/transformer-inference-arithmetic): the FLOPs-and-bytes accounting behind everything above
- [Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model"](https://people.eecs.berkeley.edu/~kubitron/courses/cs258-S08/handouts/papers/rooflinev1.pdf) (the compute-bound vs memory-bound framework)
- [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180) (vLLM): KV cache memory management
- [Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/conference/osdi22/presentation/yu): iteration-level (continuous) batching
- [Agrawal et al., "Sarathi-Serve"](https://arxiv.org/abs/2308.16369): chunked prefill
- [Zhong et al., "DistServe"](https://arxiv.org/abs/2401.09670) and [Patel et al., "Splitwise"](https://arxiv.org/abs/2311.18677): prefill/decode disaggregation
- [Dao et al., "FlashAttention"](https://arxiv.org/abs/2205.14135): why attention over long prompts doesn't become memory-bound
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192)
- [Transformer Inference Arithmetic, Part 1](/blog/2026/transformer-inference-arithmetic-part-1/) and [Part 2](/blog/2026/transformer-inference-arithmetic-part-2/): the same accounting on a 52B, four-A100 walkthrough
