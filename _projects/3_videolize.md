---
layout: page
title: Videolize
description: Text-to-video generation via spherical interpolation in Stable Diffusion latent space
importance: 3
category: Research
---

A generative system that turns a text script into a short video: scene images from **Stable Diffusion**, smooth transitions by interpolating in latent space, then synced audio from text prompts.

<a class="btn btn-sm z-depth-0" role="button" href="https://github.com/mnjkhtri/slerp-diffusion" target="_blank" rel="noopener">View on GitHub</a>

### How it works

1. **Scene frames**, generate key scenes with a Stable Diffusion pipeline.
2. **Latent interpolation**, morph between scenes with **spherical linear interpolation (slerp)** in diffusion latent space (shortest path on the hypersphere), plus **FiLM** for more coherent frame-to-frame transitions, targeting **24 FPS**.
3. **Audio**, turn audio-side prompts into tracks and mux them with the video.

Implementation lives in [mnjkhtri/slerp-diffusion](https://github.com/mnjkhtri/slerp-diffusion).

### Outcome

First place, GenAI category at **LOCUS Hack-A-Week 2023**.
