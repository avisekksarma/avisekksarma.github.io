---
layout: project
title: Topic Modeling for Nepali Political News
description: Comparative evaluation of LDA, NMF, LSA, and BERTopic on Nepali political news, published at IEEE ICICT 2025.
summary: Compared LDA, NMF, LSA, and BERTopic on a 2018–2023 corpus of Nepali political news. LDA produced more coherent topics than transformer models in this low-resource setting.
importance: 1
category: Research
year: 2025
dates: 2024–2025
venue: IEEE ICICT 2025
authors: Abhishek Sharma, Aashish Adhikari, Manoj Khatri, Hrishav Khadka, Aman Shakya
tech:
  - LDA
  - NMF
  - LSA
  - BERTopic
  - spaCy
  - gensim
paper: https://ieeexplore.ieee.org/document/11004776
---

Undergraduate research. To our knowledge this is the first comparative study of these topic models on Nepali news.

### Abstract

Identifying latent themes within large text collections is essential for understanding discourse structures across domains. This study presents a comparative evaluation of topic modeling techniques applied to Nepali political news, including Latent Dirichlet Allocation (LDA), Non-Negative Matrix Factorization (NMF), Latent Semantic Analysis (LSA), and BERTopic, a Transformer-based neural topic modeling approach. A large-scale dataset of Nepali political news articles from 2018 to 2023 was collected and preprocessed using natural language processing techniques.

Experimental results indicate that LDA outperforms neural topic modeling approaches in the Nepali language context, due to the limitations of pre-trained Transformer models for Nepali. Coherence score evaluations confirm LDA's stronger topic consistency and interpretability. The study identifies key political themes and their temporal trends, offering a view of how political discourse in Nepal shifted over the period.

The findings suggest that while neural models show promise, traditional probabilistic methods remain more effective for this low-resource language. Future work includes enhancing neural topic modeling through fine-tuned Nepali language models and expanding the dataset.
