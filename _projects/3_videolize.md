---
layout: project
title: Videolize
description: Text-to-video generation by interpolating in Stable Diffusion latent space. First place, GenAI, LOCUS Hack-A-Week 2023.
summary: Turns a written script into a short video by generating scene frames with Stable Diffusion and interpolating along the latent hypersphere.
importance: 3
category: Research
year: 2023
venue: 1st place, GenAI, LOCUS Hack-A-Week
tech:
  - PyTorch
  - Stable Diffusion
  - Gemini
github: https://github.com/mnjkhtri/slerp-diffusion
---

A generative system that maps a text script to a short video: scene images from **Stable Diffusion**, transitions by interpolating in latent space, then synced audio from text prompts.

### How it works

1. **Scene frames.** Generate key scenes with a Stable Diffusion pipeline.
2. **Latent interpolation.** Morph between scenes with spherical linear interpolation (slerp) in diffusion latent space, plus FiLM for more coherent frame-to-frame transitions, targeting 24 FPS.
3. **Audio.** Turn audio-side prompts into tracks and mux them with the video.

Implementation lives in [mnjkhtri/slerp-diffusion](https://github.com/mnjkhtri/slerp-diffusion).

### Outcome

First place, GenAI category at **LOCUS Hack-A-Week 2023**.
