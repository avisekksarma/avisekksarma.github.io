---
layout: post
title: "Counting Transformer Parameters"
subtitle: "The 12 d² shortcut, a 7B worked example, and why those weights take about 14 GB to load."
date: 2026-09-14
categories: [Tech, machine-learning, llm]
tags: [machine-learning, llm, transformers]
reading_time: 5
description: "How to count the parameters in a decoder-only Transformer: tokenizer vs embedding, 12 d_model² per block, a 7B example, and the difference between weight memory and KV-cache."
featured: true
---

When a model card says **7B**, it is counting learned numbers: the parameters. Those numbers are what you load onto a GPU before the model writes a single token. This post is a short count of where they live, why a block is about `12 d²` weights, and why a 7B model is roughly 14 GB in memory.

The stack below is the GPT-style picture from [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) and the [How Transformer LLMs Work](https://www.deeplearning.ai/courses/how-transformer-llms-work) course with Jay Alammar and Maarten Grootendorst. The counting trick is from [kipply's parameter-counting note](https://kipp.ly/p/transformer-param-count).

---

## 1. The architecture

[Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) is the 2017 **encoder–decoder** (built for translation). GPT-style LLMs drop the encoder. What you actually run is a **decoder-only** stack, the same shape as the figure you know from *How Transformer LLMs Work*: tokenizer at the **top**, `N` blocks, LM head, then `P(next token)` at the **bottom**.

<img src="/assets/img/transformer-param-count/00-decoder-stack.svg" alt="Decoder-only LLM: tokenizer and tokens at the top, embeddings, stacked blocks 1 to N, LM head, then P(next token) at the bottom. One block is masked self-attention plus an MLP." style="max-width: 36rem; width: 100%; height: auto; display: block; margin: 1.25rem auto;" />

Each block does the same two jobs, then the next block gets the result. Attention is **masked** (causal), so a token can only read earlier tokens, never the future. Residuals keep the vector width equal to `d` all the way through.

---

## 2. Three stages

The same model, left to right, as three stages.

![Decoder-only Transformer: tokenizer and embedding, one transformer block with attention and MLP, then the language-model head](/assets/img/transformer-param-count/01-architecture.svg)

**Tokenizer.** Text becomes token IDs — integers into a vocabulary. `the cat sat` might become `464, 3797, 3332`. The tokenizer does **not** create the 4096-dimensional vectors. It only picks IDs.

**Embedding.** Each ID is a lookup in a table of size `V × d`. That table is the first real parameter matrix. After this step, every token is a vector of length `d = d_model` (4096 in the example below). From here on, that width stays `d`.

**Blocks, then the LM head.** Those vectors go through the same block `N` times. Each copy has its own weights. Inside a block, **masked self-attention** lets a token read earlier tokens (not future ones), then the **MLP** processes each token by itself. Residual connections add the block's input back after each sublayer, which is why the vector size never changes.

Then a linear map `d → V` turns the **last token's** vector into one score per vocabulary entry. Softmax makes that a next-token distribution.

---

## 3. The six matrices that are almost the whole model

Open one block and almost every parameter is in six matrices.

![Attention has four d by d matrices; the MLP has d by 4d and 4d by d; together that is 12 d squared per block](/assets/img/transformer-param-count/02-parameter-count.svg)

**Attention: `4 d²`.** Four projections `Wq`, `Wk`, `Wv`, `Wo`. People talk about heads, but in the usual design

$$n_{\text{heads}} \cdot d_{\text{head}} = d$$

so each of those four matrices is `d × d`. That is `4 d²`.

**MLP: `8 d²`.** Two matrices: expand `d → 4d`, then contract `4d → d`. That is `2 × 4 d² = 8 d²`. The factor of four is the original Transformer convention. It is still the right first-order picture.

Add them:

$$\text{parameters per block} \approx 12\, d^2$$

LayerNorm and biases exist. They are `O(d)` per block, not `O(d²)`. On a 7B-class model they are a rounding error, so the `12 d²` rule can ignore them.

The embedding table is extra: `V × d`. The LM head is another `d × V`. Some models share those two (tied embeddings); some do not. Either way that is hundreds of millions, not the billions.

---

## 4. A 7B worked example

Take `d = 4096`, `N = 32` blocks, `V = 65,536`.

Per block:

$$12 \times 4096^2 = 12 \times 16{,}777{,}216 \approx 201\text{M}$$

All blocks:

$$201\text{M} \times 32 \approx 6.44\text{B}$$

Embeddings:

$$65{,}536 \times 4096 \approx 268\text{M}$$

Total ≈ **6.7B** parameters. That is a "7B model". If the LM head is not tied, add another 268M — still about 7B.

So the whole count is:

$$\text{parameters} \approx 12\, N\, d^2 + V\, d$$

Modern variants move the constant a little (grouped-query attention shrinks `Wk` and `Wv`; SwiGLU uses three MLP matrices instead of two). They do not change the picture: almost all the weights are still `O(N d²)`, plus a vocabulary table.

---

## 5. Memory: weights vs KV cache

Each parameter occupies a few bytes:

| Precision   | Bytes / param | 7B weights |
| ----------- | ------------- | ---------- |
| FP32        | 4             | ~28 GB     |
| FP16 / BF16 | 2             | ~14 GB     |
| INT8        | 1             | ~7 GB      |
| INT4        | 0.5           | ~3.5 GB    |

The number people quote for loading a 7B model in half precision:

$$7\text{B} \times 2 \text{ bytes} \approx 14\text{GB}$$

That is **static weight memory**: what you need to load the model at all.

**KV cache is separate.** During generation the model stores keys and values for tokens it has already seen, so it does not recompute attention from scratch. That memory grows with sequence length. A model that *fits* at 14 GB can still run out of memory on a long prompt. The weights do not grow as you generate; the cache does.

---

## The one number to keep

Count `12 N d² + V d`, multiply by bytes per parameter, and you know the GPU memory to **load** the LLM. Activations and the KV cache are extra, and they depend on batch size and context length — not on the parameter count on the model card.
