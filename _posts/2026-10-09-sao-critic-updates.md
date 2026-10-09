---
layout: post
title: "Are SAO's Critic Updates Really Better?"
date: 2026-10-09 17:40:00 +0800
description: "Why a lower critic loss on the current rollout batch does not necessarily mean better value estimates for the actor."
tags: [Reinforcement Learning, LLM, PPO, Critic]
categories: [Research Notes]
related_posts: false
math: true
---

## An observation about critic update order

The paper [_Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning_](https://arxiv.org/abs/2607.07508) introduces SAO, which is used in the reinforcement learning pipeline for GLM-5.2. One detail leaves me puzzled: when the critic is updated $$K$$ times, are the values used by the actor computed **before** those updates, or through an additional forward pass **after** them?

The paper's description of “Faster Value Update than Policy” attributes instability partly to inaccurate value estimates and the resulting noisy advantages. It proposes taking $$K>1$$ value-network updates per policy update, with $$K=2$$ in its experiments, so that the critic can adapt to the current policy before its estimates are used to compute advantages.

I read this as suggesting that the actor uses values computed **after the critic updates**. The discussion below assumes this ordering; it is an interpretation of the description, not a verified implementation detail.

## Revisiting the standard PPO pipeline

<figure>
  <img src="{{ '/assets/img/blog/ppo-independent-updates.png' | relative_url }}" alt="PPO data flow: old actor rollout, critic forward pass, independent actor and critic updates, and new actor rollout." style="width: 100%; height: auto; background: white;">
  <figcaption>A schematic PPO training pipeline, omitting details outside the scope of this discussion.</figcaption>
</figure>

Consider the data flow in the standard PPO pipeline illustrated above. First, the old actor generates trajectories. The critic then performs a forward pass on those trajectories. Let us call the resulting predictions **old values**. These values are used both to construct the critic's update targets and to estimate the advantages for the actor update.

After updating the critic, we could run another forward pass on the same old-actor trajectories. Let us call those predictions **new values**.

In the PPO ordering considered here, the actor uses the old values; under my interpretation of SAO, it uses the new values. Intuitively, the new values might seem more accurate because the critic has already learned from these trajectories. But is that necessarily true? To answer this, we need to revisit how the critic is trained.

## The critic loss

Critic regression is indirect. For a Monte Carlo target, the observed return is

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,
$$

but the quantity we want to predict is its **conditional expectation**:

$$
V_t^\pi:=\mathbb E_\pi[G_t\mid\mathcal H_t].
$$

Here, $$\mathcal H_t$$ denotes the information contained in the token history up to the current decision, and $$H_t$$ denotes the corresponding history supplied to the critic. Assuming square-integrable returns, the key property is

$$
V_t^\pi
=\mathbb E_\pi[G_t\mid\mathcal H_t]
=\underset{v\in\mathbb R}{\arg\min}\;
\mathbb E_\pi\!\left[(G_t-v)^2\mid\mathcal H_t\right].
$$

This motivates the critic objective

$$
\mathcal L_{\mathrm{Critic}}(\phi)
:=\mathbb E_\pi\!\left[(G_t-V_\phi(H_t))^2\right].
$$

The important point is that recovering the true value function requires minimizing the **expected** squared error, with sufficient model capacity. In practice, we cannot evaluate that expectation exactly: we optimize an estimate based on finitely many sampled trajectories.

## The problem in practice

We can repeatedly train the critic on a fixed batch of trajectories. But minimizing the loss on that batch can pull us away from the value function we actually want.

If the critic can interpolate the sampled targets, the empirical MSE is minimized by predicting each observed $$G_t$$ at its corresponding history. Consider the $$\lambda=1$$ Monte Carlo case with an undiscounted terminal reward: $$\gamma=1$$, all intermediate rewards are zero, and the terminal reward is $$R$$. Then $$G_t=R$$, so a sufficiently expressive critic can fit

$$
V_\phi(H_t)=R
$$

at every sampled position. Yet the realized reward of one continuation is generally not the conditional expected reward given its prefix. Perfectly fitting that realization is not the same as estimating the true value.

This issue was already discussed in the [GAE paper](https://arxiv.org/abs/1506.02438). Its policy update uses the value function from before the critic update to estimate advantages. The authors explain that updating the critic first can introduce additional bias: in the extreme case of overfitting, the Bellman residual

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

can vanish at every sampled timestep, making the resulting policy-gradient estimate zero.

**A lower critic loss on the same batch does not necessarily mean more accurate value estimates on that batch.** In particular, it does not by itself justify using the updated critic to compute advantages for the very trajectories it was just trained on.

## References

1. Zhenyu Hou, Yujiang Li, Jie Tang, and Yuxiao Dong. [_Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning_](https://arxiv.org/abs/2607.07508). 2026.
2. John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. [_High-Dimensional Continuous Control Using Generalized Advantage Estimation_](https://arxiv.org/abs/1506.02438). 2015.
