"""Reward interpretation for episode-based research optimizers."""

import logging
import math
from contextlib import aclosing
from numbers import Real
from statistics import fmean

from rsikit.evaluation import PolicyError

logger = logging.getLogger(__name__)


def episode_scores(episodes):
    """Successful per-seed rewards; a failed panel must never enter selection."""
    scores = {}
    for seed, episode in episodes.items():
        if episode.error is not None:
            continue
        score = (
            episode.infos[-1].get("fitness", episode.total_reward)
            if episode.infos
            else episode.total_reward
        )
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
            raise ValueError("Episode fitness must be a finite number")
        scores[seed] = float(score)
    return scores


def episode_error(episodes):
    return (
        "\n".join(f"seed={seed}: {ep.error}" for seed, ep in episodes.items() if ep.error) or None
    )


async def measure_rewards(rollouts, policies, seeds=(0,)):
    """Collect raw episodes and persist successful cumulative rewards."""
    seeds = tuple(dict.fromkeys(seeds))
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("Evaluation requires at least one seed; seed IDs must be integers")
    policies = {policy.id: policy for policy in policies}
    results = {policy_id: {} for policy_id in policies}
    for policy in policies.values():
        stored = rollouts.run.scores(policy)
        rollouts.run.save_policy(
            policy, scores={seed: None for seed in seeds if seed not in stored}
        )
    logger.info(
        "Evaluating %s episodes (%s workers)",
        len(policies) * len(seeds),
        rollouts.executor.concurrency,
        extra={"event": "evaluation_started", "total": len(policies) * len(seeds)},
    )
    async with aclosing(rollouts.collect(policies.values(), seeds=seeds)) as episodes:
        async for policy_id, seed, episode in episodes:
            results[policy_id][seed] = episode
            if episode.error is None:
                score = episode.total_reward
                rollouts.run.save_policy(policies[policy_id], scores={seed: score})
                logger.info(
                    "%s: score=%g (seed=%s)",
                    policies[policy_id].name,
                    score,
                    seed,
                    extra={"event": "policy_evaluated", "policy_id": policy_id, "seed": seed},
                )
    return results


async def mean_rewards(rollouts, policies, *, seeds=(0,)):
    """Legacy scalar callback; callers requesting only scores cannot consume errors."""
    results = await measure_rewards(rollouts, policies, seeds)
    errors = {id: error for id, episodes in results.items() if (error := episode_error(episodes))}
    if errors:
        error = PolicyError("\n".join(errors.values()))
        error.failures = errors
        raise error
    return {id: fmean(episode_scores(episodes).values()) for id, episodes in results.items()}
