---
layout: post
title: "Why Normalize GRPO Advantages? A Variance-Stabilizing Perspective"
date: 2026-08-03 12:00:00 +0800
description: "Connecting GRPO's standard-deviation normalization to a minimax variance-stabilizing transformation of binary success rates."
tags: [Reinforcement Learning, LLM, GRPO]
categories: [Research Notes]
related_posts: false
math: true
---

Among critic-free methods for LLM reinforcement learning, RLOO-style advantage estimation ([Ahmadian et al., 2024](#ref-rloo)) is a particularly clean choice: subtract a baseline without dividing by the reward standard deviation. This connects directly to the REINFORCE family of policy-gradient estimators ([Williams, 1992](#ref-reinforce)). The [DeepSeek-V3.2 technical report, Section 3.1](#ref-deepseek-v32) also uses group-mean-centered rewards without standard-deviation normalization.

The original GRPO formulation in DeepSeekMath subtracts the group mean and divides by the group standard deviation ([Shao et al., 2024](#ref-deepseekmath), Section 4.1.2). Dr. GRPO later identifies this question-dependent scaling as a source of difficulty bias and removes it ([Liu et al., 2025](#ref-dr-grpo)). Nevertheless, normalized GRPO remains an important reference point. Could keeping this normalization have useful properties of its own?

RLOO and group-mean centering should be distinguished: RLOO excludes the current sample from its baseline. For a fixed group size $$G>1$$, its centered reward is $$G/(G-1)$$ times the group-mean-centered reward. The discussion below focuses on the additional effect of dividing by the standard deviation.

Here is an intuition. Let $$p_\theta(x)$$ be the probability of success on prompt $$x$$. Our original objective is

$$
\max_\theta\; \mathbb{E}_{x\sim\mathcal{D}}[p_\theta(x)].
$$

Suppose we introduce a monotonically increasing function $$f$$ and instead optimize

$$
\max_\theta\; \mathbb{E}_{x\sim\mathcal{D}}[f(p_\theta(x))].
$$

These objectives are generally different: monotonicity preserves the ordering of success probabilities for a single prompt, but not the ordering of their averages across prompts. Still, the transformation might introduce useful statistical or optimization properties.

The following derivation connects GRPO's normalization to one such property.

## Binary rewards and the delta method

Fix a prompt $$x$$ and sample independent trajectories from the current policy. Assume that each rollout receives a binary reward,

$$
R_i\overset{\mathrm{iid}}{\sim}\operatorname{Bernoulli}(p),
\qquad p=\Pr(R_i=1).
$$

For a group of size $$G$$, the empirical success rate is

$$
\hat p_G=\frac{1}{G}\sum_{i=1}^{G}R_i.
$$

Here $$p=p_\theta(x)$$ depends on both the policy parameters and the prompt. I suppress these dependencies when they are clear from context.

For any fixed $$p\in(0,1)$$ and a function $$f$$ differentiable at $$p$$, the delta method ([van der Vaart, 1998](#ref-delta-method), Chapter 3) gives

$$
\sqrt{G}\bigl(f(\hat p_G)-f(p)\bigr)
\xrightarrow{d}
\mathcal{N}\!\left(0,\;p(1-p)[f'(p)]^2\right).
$$

Thus,

$$
V_f(p)=p(1-p)[f'(p)]^2
$$

is the asymptotic variance coefficient of the transformed estimator. It measures its local statistical noise on the $$G^{-1/2}$$ scale. If this quantity is large in some success-rate region, estimating the transformed score is relatively noisy there.

## A minimax transformation

Can we choose $$f$$ to minimize the worst-case asymptotic noise?

Consider

$$
\min_f\;\sup_{0<p<1}p(1-p)[f'(p)]^2,
$$

subject to

$$
f(0)=0,\qquad f(1)=1,\qquad f'(p)\geq 0.
$$

The endpoint constraints prevent us from making the variance arbitrarily small by simply rescaling the function. Monotonicity ensures that a higher success rate still receives a higher transformed score. We take $$f$$ to be absolutely continuous on $$[0,1]$$ and differentiable on $$(0,1)$$, allowing integrable endpoint singularities in its derivative.

Define

$$
M=\sup_{0<p<1}p(1-p)[f'(p)]^2.
$$

For finite $$M$$,

$$
p(1-p)[f'(p)]^2\leq M,
$$

and therefore

$$
f'(p)\leq\frac{\sqrt{M}}{\sqrt{p(1-p)}}.
$$

Integrating over $$[0,1]$$ yields

$$
\begin{aligned}
f(1)-f(0)
&=\int_0^1 f'(p)\,dp\\
&\leq\sqrt{M}\int_0^1\frac{dp}{\sqrt{p(1-p)}}\\
&=\pi\sqrt{M}.
\end{aligned}
$$

Since $$f(1)-f(0)=1$$, we obtain the lower bound

$$
M\geq\frac{1}{\pi^2}.
$$

This bound is attained by taking

$$
f'(p)=\frac{1}{\pi\sqrt{p(1-p)}}.
$$

Using $$f(0)=0$$, we find

$$
\boxed{f(p)=\frac{2}{\pi}\arcsin\sqrt{p}.}
$$

The factor $$1/\pi$$ is necessary to satisfy $$f(1)=1$$. For this transformation,

$$
V_f(p)=\frac{1}{\pi^2},\qquad 0<p<1.
$$

It equalizes the asymptotic variance coefficient across interior success probabilities and achieves the smallest possible worst-case value under our constraints. This is the normalized arcsine-square-root variance-stabilizing transformation. The delta-method approach to variance stabilization is classical; see [van der Vaart (1998), Chapter 3](#ref-delta-method). The argument here supplies a minimax characterization under the stated endpoint constraints.

## Connecting the transformation to GRPO

Now consider the standard normalized GRPO advantage from [DeepSeekMath](#ref-deepseekmath),

$$
A_i^{\mathrm{GRPO}}=\frac{R_i-\bar R_G}{\sigma_G},
$$

where

$$
\bar R_G=\hat p_G,\qquad
\sigma_G^2=\frac{1}{G}\sum_{j=1}^{G}(R_j-\bar R_G)^2.
$$

For binary rewards, $$\sigma_G^2=\hat p_G(1-\hat p_G)$$. For fixed $$p\in(0,1)$$, the law of large numbers and Slutsky's theorem imply, for each fixed rollout index $$i$$,

$$
A_i^{\mathrm{GRPO}}-
\frac{R_i-p}{\sqrt{p(1-p)}}
\xrightarrow{\Pr}0.
$$

We can define the advantage to be zero for all-equal reward groups; their probability vanishes as $$G\to\infty$$ for fixed interior $$p$$.

To isolate the effect of normalization, consider the resulting population-normalized, on-policy update. Ignore clipping, KL regularization, and response-length weighting, and treat advantages as fixed weights when differentiating the policy loss.

Under the usual score-function regularity assumptions, with a reward function that has no explicit dependence on $$\theta$$,

$$
\mathbb{E}_{\tau\sim\pi_\theta(\cdot\mid x)}
\left[\nabla_\theta\log\pi_\theta(\tau\mid x)\right]=0.
$$

Although $$p$$ depends on $$\theta$$, it is constant with respect to the trajectory inside this expectation. Hence,

$$
\mathbb{E}\left[p\,\nabla_\theta\log\pi_\theta(\tau\mid x)\right]=0.
$$

The population-normalized update is therefore

$$
\begin{aligned}
g_\infty(x)
&=\mathbb{E}\left[
\frac{R-p}{\sqrt{p(1-p)}}
\nabla_\theta\log\pi_\theta(\tau\mid x)
\right]\\
&=\frac{\mathbb{E}\left[R\nabla_\theta\log\pi_\theta(\tau\mid x)\right]}
{\sqrt{p(1-p)}}\\
&=\frac{\nabla_\theta p}{\sqrt{p(1-p)}}\\
&=\pi\,\nabla_\theta f(p).
\end{aligned}
$$

This is exactly the gradient of the minimax variance-stabilizing transformation, up to a constant factor. Averaging over a fixed prompt distribution gives, when differentiation and expectation can be interchanged,

$$
\mathbb{E}_{x\sim\mathcal{D}}[g_\infty(x)]
=\pi\,\nabla_\theta
\mathbb{E}_{x\sim\mathcal{D}}[f(p_\theta(x))].
$$

The factor $$\pi$$ changes the gradient magnitude but not its direction. The transformation itself, however, reweights prompts by $$1/\sqrt{p(1-p)}$$ relative to the original success-rate objective, so it can change the overall update direction across prompts. This prompt-dependent reweighting is also the issue discussed as difficulty bias by [Liu et al. (2025)](#ref-dr-grpo).

## What this interpretation establishes

The connection suggests a statistical interpretation of standard-deviation normalization: in the idealized population limit with binary rewards, GRPO's normalized update follows a transformed success-rate objective whose plug-in estimator has a minimax asymptotic variance coefficient.

There are several boundaries to this conclusion:

- **The minimized variance is that of the transformed success-rate estimator.** It is not the variance of the policy-gradient estimator, which also depends on the score vector and its relationship to rewards.
- **The delta-method argument is pointwise for interior probabilities.** It is not a uniform finite-sample guarantee near $$p=0$$ or $$p=1$$, where the derivative is singular.
- **Finite groups and practical training details matter.** Zero-variance groups, stabilizing constants in the denominator, clipping, off-policy updates, and length weighting can alter the connection.

So this does not prove that normalized GRPO is universally better, or that its advantage estimator has minimum policy-gradient variance. It does identify a concrete variance-stabilizing property of the transformed objective associated with its population-normalized update—a possible reason to study normalization beyond its departure from the original reward objective.

## References

1. <span id="ref-rloo"></span>Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. **Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs.** 2024. [arXiv:2402.14740](https://arxiv.org/abs/2402.14740). See Section 2.3 for the RLOO estimator.

2. <span id="ref-reinforce"></span>Ronald J. Williams. **Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.** _Machine Learning_, 8, 229–256, 1992. [doi:10.1007/BF00992696](https://doi.org/10.1007/BF00992696). Background on REINFORCE and expected-reward gradients.

3. <span id="ref-deepseek-v32"></span>DeepSeek-AI. **DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.** 2025. [arXiv:2512.02556](https://arxiv.org/abs/2512.02556). Section 3.1 specifies group-mean-centered advantages without division by the standard deviation.

4. <span id="ref-deepseekmath"></span>Zhihong Shao et al. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.** 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300). Section 4.1 introduces GRPO; Section 4.1.2 gives its outcome-reward normalization.

5. <span id="ref-dr-grpo"></span>Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. **Understanding R1-Zero-Like Training: A Critical Perspective.** 2025. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783). Analysis of length and question-level difficulty biases, and the Dr. GRPO alternative.

6. <span id="ref-delta-method"></span>A. W. van der Vaart. **Asymptotic Statistics.** Cambridge University Press, 1998. Chapter 3, “Delta Method.” [Chapter DOI: 10.1017/CBO9780511802256.004](https://doi.org/10.1017/CBO9780511802256.004). Background for the delta-method limit and variance-stabilizing transformations.
