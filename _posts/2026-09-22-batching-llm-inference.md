---
layout: post
title: "Batching LLM Inference"
subtitle: "How servers keep a GPU busy across many requests."
date: 2026-09-22
categories: [Tech, machine-learning, llm]
tags: [machine-learning, llm, systems, inference, batching]
reading_time: 12
description: "A GPU streams the same weights for one request or many, so batching is cheap until compute catches up. Continuous batching keeps those slots full as requests finish. KV-cache memory is usually what caps the batch."
featured: true
---

You can't serve an LLM one request at a time. That is too slow, and it leaves an expensive GPU idle. Production serving systems (vLLM, TensorRT-LLM, SGLang) exist mostly to keep the GPU busy across many requests without making any single one wait forever. The starting point is **batching**. How you batch changes the outcome a lot. Done well, you go from one request at a time to thousands of tokens per second.

## Two workers, one step

Every step of text generation produces one new token for every request currently being served. Two things happen inside the chip at once:

- **The mover** streams the model's weights from memory to the compute units.
- **The calculator** does the actual arithmetic (matrix multiplications) using those weights.

They run in parallel, so a step takes as long as whichever one is slower. That is why batching works.

For one request, computing one layer means fetching a weight matrix `W` and multiplying it by a token vector. If you stack B requests' token vectors into one matrix and multiply by the same `W`, the calculator does B times more work, but the mover fetches the same weights only once. Same trip whether you use the weights for one request or thirty.

That gives a simple cost model for one decode step:

```
step time ≈ a + b·B
```

where **a** is the (roughly fixed) cost of moving the weights, and **b** is the small extra cost each added request adds.

### Where a and b actually come from

Say the model has P parameters, stored in 16-bit precision (2 bytes each).

- **Bytes the mover must carry per step:** 2P (the weights), independent of B.
- **FLOPs the calculator must do per token:** about 2P (one multiply and one add per parameter), so 2P·B for a batch of B.

So:

```
mover time      = 2P / memory_bandwidth        (flat, doesn't depend on B)
calculator time = 2P·B / peak_FLOPs            (grows with B)
step time       = whichever is larger
```

<img src="/assets/img/batching-llm-inference/01-mover-calculator.svg" alt="Weights are read once per step by the mover; B token vectors add 2P·B FLOPs for the calculator. Step time is whichever is slower, about a + b·B." style="width: 100%; max-width: 54rem; height: auto; display: block; margin: 1.5rem auto;" />

At small B, the mover is the bottleneck and step time barely moves as B grows. Batching is nearly free. At large B, the calculator becomes the bottleneck and step time grows roughly in proportion to B. The two lines cross at a specific batch size:

```
2P / bandwidth = B* · 2P / peak_FLOPs
        →  B* = peak_FLOPs / bandwidth
```

P cancels out. The crossover point depends only on the hardware, not the model. For an A100 (≈312 TFLOPS, ≈2 TB/s bandwidth), that's B\* ≈ 312e12 / 2e12 ≈ **156**. Below that, adding requests to a batch costs almost nothing in latency. Above it, each extra request costs close to its full price.

<img src="/assets/img/batching-llm-inference/05-crossover-chart.svg" alt="Mover time is a flat line. Calculator time rises with batch size B and crosses it at B*. Step time is whichever line is higher." style="width: 100%; max-width: 48rem; height: auto; display: block; margin: 1.5rem auto;" />

The flat line dominates early. That is the free-batching region. Once the rising line overtakes it (around B\*), every extra request starts costing close to full price in step time.

**Why not just push B into the thousands?** Total throughput keeps climbing as B grows. It approaches a ceiling of `1/b` tokens/sec but never quite reaches it, since gains shrink with each doubling. Per-request speed is `1/(a + b·B)`, and that keeps falling as B gets larger. The operator sees more aggregate tokens/sec. Each user's stream gets slower. In practice you stop well before that, usually because memory runs out first (more on that below), and because nobody wants their own response crawling so the aggregate number looks better.

## The static batching problem

The naive way to batch is: collect N requests, run them together, and don't return anything until every one has finished. This is **static batching**, and it wastes GPU time.

Think of a bus that cannot let anyone off, or anyone new on, until every current passenger has reached their stop. The bus keeps driving and paying the fixed per-step cost, but seats of people who already arrived sit empty.

If a batch's requests have output lengths `L₁, L₂, ..., Lₙ`, the batch runs for `max(L)` steps, but only produces `mean(L)` steps' worth of _useful_ work per slot on average. So:

```
slot utilization = mean(L) / max(L)
```

A mix of very short and very long requests (a **bimodal** mix) is the worst case. `max(L)` stays high because of the long outliers. `mean(L)` stays low because most requests finish quickly and then occupy a dead seat. A request that arrives right after a batch starts also has to wait for the _entire_ batch to finish before it even begins. That is head-of-line blocking.

**Continuous (iteration-level) batching**, from the Orca paper, fixes this. At _every step_, the scheduler checks which requests just finished, evicts them, and admits new ones from the queue. People get on and off at every stop, not only at the end of the line.

Here's the same eight requests (four batch slots, lengths A=2, B=8, C=3, D=4 present at the start, E=3/F=2/G=4/H=2 arriving mid-flight) run both ways:

<img src="/assets/img/batching-llm-inference/02-static-batching.svg" alt="Static batching with four slots: idle red seats after A, C, and D finish, and requests E-H cannot start until B, the longest request, finishes at step 8. Twelve steps to clear eight requests." style="width: 100%; max-width: 54rem; height: auto; display: block; margin: 1.5rem auto;" />

<img src="/assets/img/batching-llm-inference/03-continuous-batching.svg" alt="Continuous batching with the same eight requests and four slots: each new request fills a seat the moment it frees, and the queue clears in eight steps." style="width: 100%; max-width: 42rem; height: auto; display: block; margin: 1.5rem auto;" />

_(Red bars are idle slots: GPU time paid for but wasted. Static batching can't start E-H until B, the longest request, finishes. Continuous batching drops each new request into a slot the moment it frees up.)_

Same eight requests, four slots, same finish order. Continuous batching clears the queue a third faster by never leaving a slot empty when work is waiting.

## Why continuous batching isn't that simple

Swapping requests in and out mid-flight sounds easy until you look at what happens _inside_ a step. The linear/MLP layers use the same shared weight matrices for every request, so they batch naturally: stack every request's token into one matrix and do one matmul. **Attention is different.** Each request looks back at its own key/value cache, and every request has a different cache length by now, since they joined the batch at different times and have generated different numbers of tokens. You can't concatenate ragged-length caches into one clean matrix operation.

Orca's answer is **selective batching**: batch what can be batched (the linear layers) and compute attention separately, per request, in the same step.

<img src="/assets/img/batching-llm-inference/04-selective-batching.svg" alt="Requests A, B, and C each contribute one new token. Linear layers and MLP batch all three against shared weights; attention runs separately on each request's own KV cache (5, 2, and 9 tokens); then linear layers batch again." style="width: 100%; max-width: 48rem; height: auto; display: block; margin: 1.5rem auto;" />

This split is what makes continuous batching possible. Every request can join or leave the batch at the linear-layer stage, because those layers don't care about sequence length. Only attention needs each request's own history, and that is handled request by request anyway.

## The limit in practice

Aggregate throughput never technically falls, and the compute crossover B\* is an inflection point, not a wall. So what actually stops a real system from cranking the batch size up?

**Memory.** Every request in the batch holds its own key/value cache, and that cache grows with every token generated. It has to sit in GPU memory the whole time the request is being served, and it is not shared between requests the way weights are. Long before compute becomes the bottleneck, the batch runs out of room to hold everyone's cache. That is why memory management (allocating, sharing, and reclaiming that cache space) becomes the next design problem once batching is solved. There is a smaller wrinkle too: a newly admitted request has to process its entire prompt in one shot on its first step (called _prefill_), which is much heavier than a normal decode step and briefly slows down every other request sharing that step. Real systems have tricks for smoothing that out.

Short version: batching is nearly free up to a point set by the ratio of your hardware's compute speed to its memory bandwidth. Past that point every added request costs close to full price in latency. In nearly every real deployment, what caps batch size isn't compute. It's running out of memory for everyone's growing cache.
