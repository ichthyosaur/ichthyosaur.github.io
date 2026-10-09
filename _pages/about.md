---
layout: about
title: About
permalink: /
subtitle: "<strong>School of Mathematical Sciences, Peking University</strong>"

profile:
  align: right
  image: jiayan-fu-avatar.jpg
  image_circular: false
  more_info: false

selected_papers: false
social: true
announcements:
  enabled: false
latest_posts:
  enabled: false
---

I am a master's student in Big Data at the School of Mathematical Sciences, Peking University, advised by [Dongyan Zhao](https://www.wict.pku.edu.cn/zhaodongyan/). Before joining PKU, I earned a bachelor's degree in Mathematics.

My research interests include Large Language Models and Reinforcement Learning, particularly understanding the mathematical mechanisms behind reinforcement learning for large language models through probability theory. Before shifting my focus to AI, my main interests in mathematics were analysis and probability theory.

My recent work mainly focuses on credit assignment in long-horizon tasks.

I am always happy to discuss research ideas in reinforcement learning for LLMs. You can reach me at [fujiayan@live.com](mailto:fujiayan@live.com).

<h2 id="selected-publications">Selected Papers</h2>

{% include selected_papers.liquid %}

<figure>
  <img src="{{ '/assets/img/publications/pact-ppo-workflow.png' | relative_url }}" alt="PPO and PACT training workflows compared." style="display: block; width: 75%; height: auto; margin-left: 0; margin-right: auto; background: white;" loading="lazy">
</figure>

We prove that three regularity conditions—Completeness, Prefix Consistency, and Neutrality—uniquely determine token-level credit, providing a unified perspective on OPD, RLOO, and GAE. We establish approximate credit sparsity under bounded outcome rewards and show how intermediate critic errors can become comparable to the underlying credit. These findings motivate Policy Aligned Critic Training (PACT), which uses an Actor-then-Critic update order and importance sampling correction to better align the critic with the updated policy. PACT achieves 72.87% average accuracy across four agentic mathematical reasoning benchmarks and 67.4% on SWE-bench Verified, outperforming GRPO and PPO in both settings.

## Selected Blogs

[Are SAO's Critic Updates Really Better?]({% post_url 2026-10-09-sao-critic-updates %}) — October 9, 2026

[Why Normalize GRPO Advantages? A Variance-Stabilizing Perspective]({% post_url 2026-08-03-grpo-variance-stabilization %}) — August 3, 2026

[Token-Mean Aggregation Is Asymptotically Unbiased]({% post_url 2026-05-27-token-mean-policy-gradient %}) — May 27, 2026

## Experience

<div class="experience-heading">
  <img src="{{ '/assets/img/rednote-icon.ico' | relative_url }}" alt="RedNote logo" width="18" height="18">
  <strong>Foundation Model Algorithm Intern · RedNote, AllSpark</strong>
</div>

April 2026 – Present

My work mainly focuses on agentic reinforcement learning, especially its algorithms.
