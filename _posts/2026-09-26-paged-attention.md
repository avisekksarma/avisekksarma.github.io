---
layout: post
title: "PagedAttention: How vLLM Manages KV-Cache Memory Like an Operating System"
subtitle: "Why contiguous KV reservations waste memory, and how paging gets 2–4× more throughput."
date: 2026-09-26
categories: [Tech, machine-learning, llm]
tags: [machine-learning, llm, systems, inference, kv-cache]
reading_time: 16
description: "Serving an LLM to many users is mostly a KV-cache memory problem. vLLM borrowed paging from operating systems so the cache grows in small blocks instead of worst-case chunks, which is why throughput goes up 2–4×."
featured: true
---

Serving a large language model to many users at once is mostly a memory problem. The GPU can only run as many requests together as it can hold in memory, and the thing that eats that memory isn't only the model weights — it's the **KV cache**, a per-request structure that grows with every token generated. Before 2023, serving systems managed this memory in a simple but very wasteful way. The vLLM paper (_"Efficient Memory Management for Large Language Model Serving with PagedAttention"_, SOSP 2023) fixed it by borrowing an idea operating systems have used for decades: **paging**.

This note walks through why the old approach wastes so much memory, how paging fixes it, what goes wrong when memory runs out, and why all of this translates into 2–4× more throughput.

---

## 1. A quick refresher: what the KV cache is

LLMs generate text one token at a time. To produce each new token, the attention mechanism compares the new token against **every previous token**, using each previous token's _Key_ (K) and _Value_ (V) vectors. Recomputing K and V for all previous tokens at every step would be hugely wasteful, so systems compute them once and **cache** them. Each step then only computes K and V for the one new token and appends them to the cache.

Two phases are worth naming, since they come up later:

- **Prefill:** the whole prompt is processed in one forward pass, all tokens in parallel. This fills the cache for the prompt.
- **Decode:** after that, the model produces one token per forward pass, sequentially, appending one token's K/V to the cache each time.

## 2. How big is the KV cache?

At every layer, each token produces one Key vector and one Value vector, each `d_model` numbers long. The cache stores both, for every past token, at every layer:

```
KV bytes per token = 2 (K and V) × n_layers × d_model × bytes_per_number
```

Two examples:

- **OPT-13B** (the vLLM paper's running example): 2 × 40 × 5120 × 2 bytes ≈ **800 KB per token**. One request at the maximum length of 2048 tokens needs about **1.6 GB**.
- **GPT-2 small**: 2 × 12 × 768 × 2 ≈ **36 KB per token**.

Many recent models (Llama, Qwen, etc.) use grouped-query attention, where only a few heads store K/V. For those, replace `d_model` with `n_kv_heads × head_dim`, which is much smaller.

To see why this matters: in the paper's setup, a 13B model on a 40 GB A100 uses about 26 GB for weights, leaving roughly 12 GB for KV cache — about **15,000 tokens** of cache. If each request reserved room for the full 2048 tokens, only about **7 requests** could run at once. Since serving throughput grows with how many requests run together (section 7 shows why), KV memory directly limits throughput.

Two facts about the cache cause all the trouble that follows:

1. It **grows by one token's worth every decode step**.
2. Its **final size is unknown** until the model emits its end-of-sequence token.

## 3. The old way: one contiguous chunk per request

Systems before vLLM (Orca, NVIDIA FasterTransformer) stored each request's KV cache as **one contiguous chunk of memory**, because attention kernels were written for plain contiguous tensors. And since the final length is unknown, that chunk has to be sized for the **worst case** — the maximum possible length.

This wastes memory in three separate ways. Here's a small memory region holding three requests, each with a 12-slot contiguous reservation:

<div class="kv-figure" markdown="0" style="width: 100%; max-width: 48rem; margin: 1.5rem auto;">
{% include_relative paged-attention/01-contiguous-allocation.svg %}
</div>

Free memory in total is 7 + 5 = **12 slots** — exactly enough for the new request. But the largest single gap is only 7, so **the request is rejected** even though the memory technically exists.

The three kinds of waste:

- **Reserved (amber):** slots for tokens the request _will_ eventually generate, but which sit empty for most of its lifetime. A request that ends at token 500 has been holding slot 500 since step 1.
- **Internal fragmentation (coral):** slots beyond the request's actual final length. These are never used — they exist only because the length had to be guessed as the maximum.
- **External fragmentation (dashed):** free memory broken into gaps between chunks. There's enough in total, but no single gap is big enough.

When the vLLM authors measured existing systems, only about **20–40% of KV-cache memory actually held token data**. The rest was waste of these three kinds.

## 4. The fix: paging, borrowed from operating systems

Operating systems solved this exact problem for program memory long ago. A program sees one clean, contiguous address space (_virtual memory_). Underneath, the OS cuts physical RAM into fixed-size **pages**, places them wherever there's room, and keeps a **page table** mapping "the program's page 3" to "physical frame 812". The program never knows its memory is scattered.

PagedAttention applies the same idea to the KV cache:

| OS virtual memory              | vLLM                                     |
| ------------------------------ | ---------------------------------------- |
| Process                        | Request (sequence)                       |
| Byte                           | One token's K/V                          |
| Page (e.g. 4 KB)               | KV block (e.g. 16 tokens)                |
| Page table                     | Block table                              |
| Physical frame                 | Physical KV block in GPU memory          |
| Allocate a page on first touch | Allocate a block when the last one fills |

Here are two requests with a block size of 4 tokens. Each request's **block table** maps its _logical_ blocks (their order in the sequence) to _physical_ blocks (where they actually live in GPU memory):

<div class="kv-figure" markdown="0" style="width: 100%; max-width: 48rem; margin: 1.5rem auto;">
{% include_relative paged-attention/02-paging-block-tables.svg %}
</div>

### How PagedAttention works

The figure already has the whole mechanism. Attention still needs every previous token's K and V, but it never needed those vectors to sit in one contiguous chunk. It needed a way to visit them in sequence order. That is the block table: logical block 0, then 1, then 2, regardless of where each one lives in GPU memory.

For request A, the table says 0 → #7, 1 → #1, 2 → #3. To attend, you walk that list: tokens 0–3 from physical block #7, tokens 4–7 from #1, tokens 8–9 from #3 (and ignore the empty slots in the last block). Same tokens, same order, same answer. Only the addresses changed.

Writing is the same map in reverse. A new token goes in the next empty slot of the last logical block. If that block is full, you take any free physical block, append it to the table, and write there. Prefill does this once for the prompt; decode grows one slot at a time. vLLM's default block is 16 tokens: small enough that a half-empty last block is cheap, large enough that you are not looking up a new block for every token.

Each kind of waste from section 3 is handled separately:

- **Reserved → gone.** A block is allocated only when the previous one fills, so no memory is held for future tokens.
- **Internal fragmentation → bounded.** Only the last block can be partly empty, so each request wastes fewer than `block_size` slots — at most 15 with 16-token blocks, versus up to ~2,000 with worst-case reservation. The paper reports total waste under about 4%.
- **External fragmentation → gone.** All blocks are the same size, so any free block fits any request. "Enough free memory in total" and "can admit" become the same thing.

### Walkthrough: one request, block size 4

A request with a 6-token prompt that generates 5 tokens:

| Step     | Tokens cached | Blocks held | Empty slots | What happens                 |
| -------- | ------------- | ----------- | ----------- | ---------------------------- |
| Prefill  | 6             | 2           | 2           | allocate 2 blocks (⌈6/4⌉)    |
| Decode 1 | 7             | 2           | 1           | fits in block 2              |
| Decode 2 | 8             | 2           | 0           | block 2 now full             |
| Decode 3 | 9             | 3           | 3           | **allocate a new block**     |
| Decode 4 | 10            | 3           | 2           |                              |
| Decode 5 | 11            | 3           | 1           | done → **free all 3 blocks** |

The general rule is `blocks = ⌈tokens / block_size⌉` and `waste = blocks × block_size − tokens < block_size`. With contiguous reservation at a max length of 16, this same request would have held 16 slots from the first step to the last.

## 5. Bonus: sharing blocks, like `fork()`

Once memory goes through a table, two sequences can point at the _same_ physical block. Suppose you ask for 2 samples from one 7-token prompt (parallel sampling), or run beam search. The prompt's KV cache is identical for every sample, so it's stored once: both samples start with the table `[#7, #1]`, and each of those blocks has a **reference count** of 2.

Block #7 holds prompt tokens 1–4 and is full. Block #1 holds prompt tokens 5–7 and still has one free slot. When sample 1 generates its first token, it has to write into #1 — but #1 is shared, and sample 2's token must not end up there too. So sample 1 first gets a **private copy** of #1 (block #3), writes into the copy, and #1's reference count drops to 1. Sample 2 then writes into the original #1 directly, since it is now the only owner:

<div class="kv-figure" markdown="0" style="width: 100%; max-width: 48rem; margin: 1.5rem auto;">
{% include_relative paged-attention/04-copy-on-write.svg %}
</div>

This is **copy-on-write**, exactly what an OS does when `fork()` creates a child process: parent and child share memory pages until one of them writes, and only the written page is copied. A physical block is freed only when its reference count drops to 0.

## 6. The catch: running out of blocks

### The question at the root of it

When a request arrives, you know its prompt length. You **don't** know how many tokens it will generate — it might stop after 5 or run to the maximum. Yet every generated token needs KV memory. So at admission time the server has to decide how much memory to promise a request whose final size it can't know. There are two philosophies.

**Pessimistic (Orca-style):** "Assume the worst." Reserve memory for the maximum length at admission. A running request can then never run out of memory mid-generation. The cost is that most of the reservation sits empty, so few requests fit.

**Optimistic (vLLM):** "Give it what it needs right now." Allocate blocks only for the prompt at admission, then add one block at a time as it grows. Many more requests fit. The cost is that you've effectively promised memory you may not have later.

### A running example: 10 blocks of memory

Say the GPU has room for **10 KV blocks**, and the maximum request length is **4 blocks**.

_Pessimistic:_ each request reserves 4 blocks, so only **2 requests** are admitted (8 blocks). The 2 leftover blocks can't hold a third request, which would need 4. Early on, those 2 requests each use only 1–2 of their 4 blocks — most of the memory holds nothing.

_Optimistic:_ each request takes only what its prompt needs, so **5 requests** are admitted. A batch of 5 instead of 2 means much more throughput. But watch what happens as they all keep growing:

<div class="kv-figure" markdown="0" style="width: 100%; max-width: 48rem; margin: 1.5rem auto;">
{% include_relative paged-attention/05-preemption.svg %}
</div>

(Blocks are scattered in memory, as paging allows — which is why each request's blocks aren't next to each other.)

### Why someone has to be evicted

At moment 2, C can't produce its next token — there's nowhere to store that token's K and V. Could C just pause until another request finishes and frees memory? Sometimes. But every running request keeps growing, so soon A, B, and D will each fill their current block too. If they all pause waiting for memory that only they themselves could free, nobody finishes and nothing is ever freed. So the system has to force space open by removing a request. That removal is **preemption**, and vLLM answers three questions about it.

**1. Who gets removed? The newest request.** vLLM serves requests first-come-first-served. The earliest arrivals have waited longest, so removing the latest arrival keeps things fair. Removing the oldest request would punish the one that has been in the system longest.

**2. How much of it? All of its blocks at once.** Why not take just one of E's blocks? Because to generate its next token, E's attention needs the K and V of _every_ previous token (the walk through the block table in section 4). With even one block missing, E can't take a single step — it would be stuck while still holding its other blocks, the worst of both outcomes. So eviction is **all-or-nothing**: a request either keeps everything or gives everything up.

**3. What happens to the evicted request?** It goes back to the _front_ of the waiting queue, ahead of brand-new requests. Its generated text is kept — those are just token IDs, a few bytes each. Only the large derived data, its KV cache, is lost. When memory frees up, it comes back in one of two ways.

### Two ways to bring it back

**Swap:** before freeing the victim's blocks, copy them from GPU memory to CPU RAM; copy them back when it resumes. The cost is transfer time. For OPT-13B, 500 tokens × 800 KB ≈ 400 MB; over a PCIe link of roughly 25 GB/s, that's about 16 ms each way. The data survives, but it needs spare CPU memory and many small block-sized transfers.

**Recompute:** simply throw the KV cache away. When the request resumes, treat "original prompt + everything it generated so far" as one long new prompt and run **prefill** on it. That rebuilds the exact same KV cache.

Recompute sounds wasteful ("redo all that work?"), but it's far cheaper than the original generation. The reason is the difference between decode and prefill. A decode step is **memory-bound**: each step must read all the model weights from memory just to produce one token per request, so its cost is mostly that fixed weight-read time (say ~20 ms). Prefill processes all its tokens in one pass: the weights are read once, and the extra cost is just the arithmetic per token (say ~0.5 ms per token).

Say the evicted request had 100 prompt tokens and had generated 400 tokens:

- **Originally:** those 400 tokens took 400 sequential decode steps ≈ 400 × 20 ms ≈ **8 seconds**.
- **Recompute:** one prefill pass over 500 tokens ≈ 20 ms + 500 × 0.5 ms ≈ **0.27 seconds**.

Generation was slow because it was sequential; rebuilding the cache happens all at once. So rebuilding costs roughly 30× less than creating it the first time. The paper found the two methods roughly comparable at typical block sizes — swap gets worse with small blocks (many tiny transfers), while recompute's cost doesn't depend on block size at all.

(Side note: on hardware where CPU and GPU share the same physical memory, such as Apple Silicon, "swapping to CPU" has little meaning — recompute is the natural choice there.)

### Why the optimistic bet usually wins — and when it doesn't

Preemption is the price of optimism. vLLM still takes the bet because most requests end far below the maximum length. Pessimistic reservation pays for the worst case on _every_ request, _all the time_. Optimistic allocation pays only occasionally, when many requests happen to grow at once.

There is a failure mode, though. If memory is very tight, the system can fall into a loop: preempt a request, readmit it, run out again, preempt it again — spending more and more time rebuilding caches instead of generating new tokens. Operating systems call this **thrashing**, and throughput collapses when it happens.

In one line: **pessimistic = safe but wasteful every time; optimistic = efficient on average, with occasional preemption costs that can turn into thrashing under severe memory pressure.**

## 7. Results: why less waste means more throughput

The chain of cause and effect is:

**less wasted memory → more requests fit → bigger batch → more throughput**

The last link needs one fact about GPU inference. Each decode step has a big **fixed cost** `a` (reading all the model weights from memory) plus a small **per-request cost** `b`. Because the weights are read once per step and shared by every request in the batch, one step for a batch of `B` requests takes roughly:

```
step time ≈ a + b·B
throughput ≈ B / (a + b·B)   tokens per second
```

As `B` grows, the fixed cost is spread over more requests, so throughput rises.

Now plug in numbers. With about 12 GB for KV cache at 800 KB per token, there's room for roughly 15,000 tokens:

- **Pessimistic:** every request reserves 2,048 tokens → 15,000 / 2,048 ≈ **7 requests** in the batch.
- **Paged:** requests hold only what they actually use. If a typical request ends up around 500 tokens (prompt + output) → 15,000 / 500 ≈ **30 requests** in the batch.

With illustrative values `a = 20 ms` and `b = 0.5 ms`:

- B = 7: 7 / 23.5 ms ≈ **300 tokens/sec**
- B = 30: 30 / 35 ms ≈ **860 tokens/sec**

That's about **2.9×**, from memory bookkeeping alone. Same model, same arithmetic. That is the mechanism behind the paper's reported **2–4× higher throughput** than FasterTransformer and Orca, with larger gains for long sequences and for parallel sampling or beam search, where block sharing helps most.

### What "at the same latency" means

Serving papers don't just report maximum tokens/sec, because you could get that by letting users wait forever. Instead, they run each system at increasing request rates and measure latency per output token.

Every server has a **capacity** — the rate at which it can finish requests. While requests arrive slower than that, latency stays low. As the arrival rate approaches capacity, the waiting queue starts to grow and latency rises sharply; past capacity, the queue grows without bound. A system that fits bigger batches has higher capacity, so its curve bends upward later:

<div class="kv-figure" markdown="0" style="width: 100%; max-width: 48rem; margin: 1.5rem auto;">
{% include_relative paged-attention/06-latency-vs-rate.svg %}
</div>

To read it: pick the latency you're willing to accept (the dashed line) and see how much traffic each system handles before crossing it. Here the contiguous system manages 0.6 requests/sec and the paged one 1.5 — **2.5× more load at the same latency**. That's what "2–4× throughput at the same latency" means. (The numbers here are made up to show the shape; the paper's real plots have the same form.)

One interesting detail in the paper's comparison: the authors built several versions of Orca-style reservation, including an **"oracle"** version that magically knows each request's exact final length and reserves exactly that. vLLM _still_ beats it. Even perfect knowledge doesn't remove the reserved-for-future waste (memory held from step 1 for tokens that arrive much later) or external fragmentation. Blocks fix both — so the gain comes from the memory layout itself, not just from better guessing.

## 8. What a simplified block manager needs

If you were to build a toy version of this, it has three parts.

**State it keeps:**

- a **free list** of unused block IDs out of a fixed budget of N blocks;
- for each running request, a **block table** (list of block IDs) and its current token count.

**Events it handles every step:**

- **Admission:** a new request wants to join → are there enough free blocks for its prompt?
- **Growth:** a running request just filled its last block → give it one more.
- **Finish:** a request completes → return all its blocks to the free list.
- **Crunch:** a request needs a block and none are free → someone must be preempted.

**Decisions that are policy, not mechanics:**

- **Admission rule:** fully optimistic ("prompt fits now"), or keep some headroom in reserve to reduce preemptions?
- **Victim choice:** newest arrival (as vLLM does), or something else — fewest blocks, lowest priority, least progress?
- **Recovery:** swap, recompute, or simply reject the request?

The first two groups are bookkeeping. The third group is where the interesting tradeoffs live: each choice shifts the balance between throughput, tail latency, and fairness, especially under tight memory.

## 9. Summary

- The KV cache grows one token at a time and its final size is unknown, so older systems reserved worst-case contiguous memory per request, wasting 60–80% of it through reservation, internal fragmentation, and external fragmentation.
- PagedAttention splits KV memory into fixed-size blocks, maps each request's logical blocks to scattered physical blocks through a block table (exactly like OS page tables), and allocates blocks only as needed. Waste drops below ~4%.
- Attention still works because the block table is the sequence order: walk logical blocks 0, 1, 2, …, look up each physical block, and you visit the same tokens you always did. A new token goes in the next slot; a new block is allocated only when the last one fills.
- Block tables also enable sharing, with reference counts and copy-on-write, for parallel sampling and beam search.
- On-demand allocation is optimistic, so running out of blocks is possible; vLLM preempts the newest request entirely and later restores it by swapping or recomputing (cheap, because prefill is parallel).
- Less waste → bigger batches → 2–4× throughput at the same latency.

{% include quiz.html quiz="pagedattention" %}
