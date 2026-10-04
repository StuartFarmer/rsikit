"""Reward-based fitness conventions shared by the control-policy experiments."""

import logging
from contextlib import aclosing
from statistics import fmean

from rsikit.evaluation import Measurement as Measurement
from rsikit.evaluation import PolicyError

logger = logging.getLogger(__name__)


async def measure_rewards(rollouts, policies, seeds=(0,)):
    """Choose cumulative reward as fitness; persist these explicit optimizer measurements."""
    measured, _ = await _measure_rewards(rollouts, policies, seeds)
    return measured


async def _measure_rewards(rollouts, policies, seeds):
    seeds = tuple(dict.fromkeys(seeds))
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("Evaluation requires at least one seed; seed IDs must be integers")
    policies = {policy.id: policy for policy in policies}
    measured = {policy_id: {} for policy_id in policies}
    error = None
    failures = {}
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
    try:
        async with aclosing(rollouts.collect(policies.values(), seeds=seeds)) as episodes:
            async for policy_id, seed, episode in episodes:
                score = episode.total_reward
                measured[policy_id][seed] = score
                rollouts.run.save_policy(policies[policy_id], scores={seed: score})
                logger.info(
                    "%s: score=%g (seed=%s)",
                    policies[policy_id].name,
                    score,
                    seed,
                    extra={"event": "policy_evaluated", "policy_id": policy_id, "seed": seed},
                )
    except PolicyError as exc:
        if not exc.failures or not set(exc.failures) <= policies.keys():
            raise
        error = exc
        failures = exc.failures
    return {
        policy_id: Measurement(scores, failure=failures.get(policy_id))
        for policy_id, scores in measured.items()
    }, error


async def mean_rewards(rollouts, policies, *, seeds=(0,)):
    """Fitness callback for optimizers that consume one mean return per candidate."""
    measured, error = await _measure_rewards(rollouts, policies, seeds)
    if error is not None:
        raise error
    return {key: fmean(value.scores.values()) for key, value in measured.items()}
