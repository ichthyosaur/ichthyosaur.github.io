---
layout: post
title: "Why Token-Mean Aggregation Is Asymptotically Unbiased"
date: 2026-05-27 12:00:00 +0800
description: "How the law of large numbers explains the asymptotic unbiasedness of token-mean aggregation, up to a positive gradient scale."
tags: [Reinforcement Learning, LLM, DAPO, GRPO]
categories: [Research Notes]
related_posts: false
math: true
---

One change introduced by DAPO is to replace GRPO's sequence-mean-token-mean reduction with a token-level mean: sum the token losses across responses, then divide by the total number of tokens ([Yu et al., 2025](#ref-dapo), Section 3.3). The original GRPO objective instead averages within each response before averaging across responses ([Shao et al., 2024](#ref-grpo), Section 4.1).

Why should token-mean aggregation better preserve the policy-gradient direction? Its denominator is random, so the answer is not immediately obvious.

The key trick is simple: divide both the numerator and denominator by the number of rollouts. The law of large numbers then turns the ratio into an expected gradient divided by an expected length.

**Token-mean aggregation is asymptotically unbiased for the policy gradient up to a positive scale factor**, under the integrability conditions stated below. The following derivation explains why.

## The gradient we want to estimate

Fix a prompt $$x$$ and a policy parameter $$\theta$$. Let a response be $$\tau$$, with length $$L=\lvert\tau\rvert\geq 1$$, and write its probability as $$p_\theta(\tau\mid x)$$. The state $$s_t$$ includes the prompt and preceding tokens. The trajectory includes all sampled actions, including termination where applicable.

Our objective is the expected sequence reward,

$$
J_x(\theta)=\mathbb{E}_{\tau\sim p_\theta(\cdot\mid x)}[R(\tau)].
$$

Assume the reward has no explicit parameter dependence and the usual conditions for interchanging differentiation and expectation hold. The score-function identity underlying REINFORCE ([Williams, 1992](#ref-reinforce)) gives

$$
\begin{aligned}
\nabla_\theta J_x(\theta)
&=\sum_\tau R(\tau)\nabla_\theta p_\theta(\tau\mid x)\\
&=\mathbb{E}\!\left[R(\tau)\nabla_\theta\log p_\theta(\tau\mid x)\right]\\
&=\mathbb{E}\!\left[R(\tau)\sum_{t=1}^{L}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
\end{aligned}
$$

We can subtract a response-independent baseline $$b(x)$$ because the expected trajectory score is zero. With $$A(\tau)=R(\tau)-b(x)$$, define the response's **total token-gradient contribution** as

$$
S(\tau)=A(\tau)\sum_{t=1}^{L}\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

Then

$$
\boxed{\nabla_\theta J_x(\theta)=\mathbb{E}[S].}
$$

The sequence-level advantage is held fixed when differentiating the surrogate loss. An arbitrary normalized or sample-dependent advantage need not satisfy this identity; the baseline assumption here makes the target precise. I also set aside clipping, off-policy updates, and regularization to isolate aggregation. All expressions below use the gradient-ascent convention.

## Token-mean: a ratio of sample averages

Sample $$N$$ independent responses at the fixed policy, with contributions $$S_i=S(\tau_i)$$ and lengths $$L_i$$. Token-mean aggregation gives

$$
\widehat g_{\mathrm{TM}}
=\frac{\sum_{i=1}^{N}S_i}{\sum_{i=1}^{N}L_i}
=\frac{\frac{1}{N}\sum_{i=1}^{N}S_i}
{\frac{1}{N}\sum_{i=1}^{N}L_i}.
$$

Assume $$\mathbb{E}\|S\|<\infty$$ and $$0<\mathbb{E}[L]<\infty$$. By the strong law of large numbers,

$$
\frac{1}{N}\sum_{i=1}^{N}S_i\xrightarrow{\mathrm{a.s.}}\mathbb{E}[S],
\qquad
\frac{1}{N}\sum_{i=1}^{N}L_i\xrightarrow{\mathrm{a.s.}}\mathbb{E}[L].
$$

Continuity of division at a positive denominator therefore yields

$$
\boxed{
\widehat g_{\mathrm{TM}}
\xrightarrow{\mathrm{a.s.}}
\frac{\mathbb{E}[S]}{\mathbb{E}[L]}
=\frac{1}{\mathbb{E}[L]}\nabla_\theta J_x(\theta).
}
$$

Using the weak law and Slutsky's theorem gives the corresponding convergence-in-probability statement; see [van der Vaart (1998), Chapter 2](#ref-convergence).

At a fixed policy, $$1/\mathbb{E}[L]$$ is a deterministic positive scalar. Whenever the true gradient is nonzero, the limiting update points in exactly the same direction. The random denominator has become a scale factor in the limit.

There is also a useful finite-batch observation:

$$
\widehat g_{\mathrm{TM}}
=\frac{N}{\sum_i L_i}\left(\frac{1}{N}\sum_i S_i\right).
$$

For any realized batch, token-mean has exactly the same direction as that batch's sequence-averaged **sum** of token gradients. It scales the whole batch together instead of weighting individual responses by their inverse lengths.

## Asymptotic unbiasedness

To express the result in terms of expected gradients, assume additionally that $$\mathbb{E}\|S\|^2<\infty$$. Since $$L_i\geq 1$$,

$$
\|\widehat g_{\mathrm{TM}}\|^2
\leq \left(\frac{1}{N}\sum_{i=1}^N\|S_i\|\right)^2
\leq \frac{1}{N}\sum_{i=1}^N\|S_i\|^2.
$$

Thus the estimators have uniformly bounded second moments and are uniformly integrable. Together with the convergence above, this gives

$$
\boxed{
\lim_{N\to\infty}\mathbb{E}[\widehat g_{\mathrm{TM}}]
=\frac{1}{\mathbb{E}[L]}\nabla_\theta J_x(\theta).
}
$$

This is the sense in which **token-mean is asymptotically unbiased**: its expected update approaches the true policy gradient multiplied by the positive scalar $$1/\mathbb{E}[L]$$. This scalar changes the gradient magnitude without changing its direction. Finite batches can still have ratio-estimation bias.

## Where sequence-mean-token-mean changes the target

Averaging token gradients within each response first gives

$$
\widehat g_{\mathrm{SM}}
=\frac{1}{N}\sum_{i=1}^{N}\frac{S_i}{L_i}
\xrightarrow{\mathrm{a.s.}}\mathbb{E}\!\left[\frac{S}{L}\right].
$$

In general,

$$
\mathbb{E}\!\left[\frac{S}{L}\right]
\neq\frac{\mathbb{E}[S]}{\mathbb{E}[L]},
$$

and the two vectors need not even be parallel. Each response's contribution is reweighted by its own inverse length. When length and gradient contribution are related, this can change the update direction even with infinitely many samples. This is the response-length weighting problem analyzed by [Liu et al. (2025), Section 3.1](#ref-dr-grpo).

## Scope of the conclusion

The argument supports a precise claim: **a shared token-count denominator preserves the underlying population gradient direction in the large-sample limit, provided the numerator already estimates that gradient.** It does not make every part of GRPO or DAPO unbiased.

Two details matter when applying this to training:

- **Expected length depends on the policy.** The scale $$1/\mathbb{E}_\theta[L]$$ is constant with respect to rollout randomness at a fixed $$\theta$$, but can change during training. The update derived here is not generally $$\nabla_\theta(J_x/\mathbb{E}_\theta[L])$$, which would include a derivative of expected length.
- **The scope of the denominator matters.** With independent prompt-response pairs from a fixed prompt distribution and one shared batch denominator, the same reasoning gives $$\mathbb{E}_{x,\tau}[S]/\mathbb{E}_{x,\tau}[L]$$. Normalizing each prompt separately and then averaging instead gives $$\mathbb{E}_x[\nabla J_x/\mathbb{E}[L\mid x]]$$ in the per-prompt large-group limit, which can reweight prompts and change the overall direction.

Group-normalized advantages, dynamic sampling, and clipped surrogate losses introduce further effects beyond this calculation. The useful insight is specifically about aggregation: replacing individual inverse-length weights with one shared denominator removes that source of relative response weighting, with asymptotic unbiasedness up to a positive scale under the stated assumptions.

## References

1. <span id="ref-dapo"></span>Qiying Yu et al. **DAPO: An Open-Source LLM Reinforcement Learning System at Scale.** 2025. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476). Section 3.3 introduces the token-level policy-gradient loss.

2. <span id="ref-grpo"></span>Zhihong Shao et al. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.** 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300). Section 4.1 presents the original GRPO objective.

3. <span id="ref-reinforce"></span>Ronald J. Williams. **Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.** _Machine Learning_, 8, 229–256, 1992. [doi:10.1007/BF00992696](https://doi.org/10.1007/BF00992696).

4. <span id="ref-convergence"></span>A. W. van der Vaart. **Asymptotic Statistics.** Cambridge University Press, 1998. [Chapter 2, “Stochastic Convergence”](https://www.cambridge.org/core/books/abs/asymptotic-statistics/stochastic-convergence/106715490302CD0E35765E8BDE5CE514). Background on stochastic convergence, continuous mappings, and Slutsky's lemma.

5. <span id="ref-dr-grpo"></span>Zichen Liu et al. **Understanding R1-Zero-Like Training: A Critical Perspective.** 2025. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783). Sections 3.1–3.2 discuss length bias and fixed normalization in Dr. GRPO.
