"""Reflective policy search with instance coverage, adapted from gepa-ai/gepa.

Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors. See LICENSE.
"""

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from statistics import fmean
from types import SimpleNamespace

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from slick import prompt

from rsikit import PolicyDefinition
from rsikit.evaluation import episode_error, episode_scores
from rsikit.optimization import SequentialOptimizer, search, validate_results
from rsikit.policy import InvalidPolicy


class Proposal(BaseModel, extra="forbid"):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    implementation: str = Field(min_length=1)


@dataclass(frozen=True)
class Candidate:
    policy: PolicyDefinition
    scores: tuple[float, ...]
    parents: tuple[int, ...] = ()

    @property
    def score(self):
        return fmean(self.scores)


@dataclass
class Result:
    population: list[Candidate] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    rollouts: int = 0
    reflection_calls: int = 0


def pareto_weights(scores):
    """Prune redundant instance-win coverage and count the retained wins."""
    fronts = []
    for column in zip(*scores):
        best = max(column)
        fronts.append({i for i, score in enumerate(column) if score == best})
    order = dict.fromkeys(i for front in fronts for i in sorted(front))
    remaining = set(order)
    for i in sorted(order, key=lambda i: fmean(scores[i])):
        wins = [front for front in fronts if i in front]
        if all((front & remaining) - {i} for front in wins):
            remaining.remove(i)
    return dict(Counter(i for front in fronts for i in sorted(front) if i in remaining))


class GEPA(SequentialOptimizer):
    """Seeds are instances; selection-panel evidence never enters reflection prompts."""

    def __init__(
        self,
        task,
        provider,
        evaluate,
        trace,
        *,
        initial_policy=None,
        train_seeds=(0, 1, 2),
        selection_seeds=(10, 11, 12),
        budget=1000,
        minibatch_size=3,
        seed=0,
        on_event=None,
    ):
        super().__init__()
        self.initial_policy = initial_policy
        self.train, self.selection = tuple(train_seeds), tuple(selection_seeds)
        self.budget, self.minibatch_size, self.seed = budget, minibatch_size, seed
        self.task, self.provider, self.evaluate, self.trace = task, provider, evaluate, trace
        self.on_event = on_event
        self.result = Result()
        self.training_evidence = []

    @property
    def best(self):
        best = max(self.result.population, key=lambda row: row.score, default=None)
        return best.policy if best else None

    def emit(self, kind, data):
        if self.on_event:
            self.on_event(kind, data)

    @prompt(template="reflect.j2", output_type=Proposal)
    async def reflect(self, policy, examples, *, generated: Proposal) -> PolicyDefinition:
        try:
            policy = PolicyDefinition.from_text(
                generated.implementation, name=generated.name, description=generated.description
            )
        except ValueError as exc:
            raise InvalidPolicy(str(exc)) from exc
        policy.validate()
        return policy

    async def assess(self, policy, seeds):
        if self.result.rollouts + len(seeds) > self.budget:
            raise RuntimeError("Insufficient budget for complete panel")
        self.result.rollouts += len(seeds)
        results = await self.evaluate([policy], seeds=seeds)
        validate_results(results, [policy.id], seed_panel=set(seeds))
        measured = results[policy.id]
        self.training_evidence.append({policy.id: dict(measured)})
        scores = episode_scores(measured)
        error = episode_error(measured) or (None if scores else "No accepted episodes")
        self.emit(
            "measurement",
            dict(
                policy=policy.to_text(),
                seeds=list(seeds),
                scores=scores,
                failure=error,
                accepted=error is None,
            ),
        )
        if error is not None:
            return None
        if scores.keys() != set(seeds):
            raise ValueError("Evaluation returned an incomplete or unexpected seed panel")
        return tuple(scores[seed] for seed in seeds)

    def examples(self, policy, seeds):
        examples = []
        for seed in seeds:
            episode = self.trace(policy, seed)
            if episode is None:
                raise ValueError("Training episode evidence is missing")
            evidence = dict(
                observations=episode.observations[-9:],
                actions=episode.actions[-8:],
                rewards=episode.rewards[-8:],
                terminations=episode.terminations[-8:],
                truncations=episode.truncations[-8:],
            )
            rendered = json.dumps(
                evidence,
                ensure_ascii=False,
                default=lambda value: (
                    value.tolist() if isinstance(value, (np.ndarray, np.generic)) else repr(value)
                ),
            ).encode()
            truncated = len(rendered) > 16384
            examples.append(
                dict(
                    seed=seed,
                    reward=episode.total_reward,
                    steps=len(episode),
                    terminated=episode.terminations[-1] if episode.terminations else False,
                    truncated=episode.truncations[-1] if episode.truncations else False,
                    evidence=rendered[:16384].decode(errors="ignore"),
                    evidence_truncated=truncated,
                )
            )
        return examples

    async def mutate(self, parent_id, seeds):
        parent = self.result.population[parent_id]
        record = dict(parent=parent_id, train_seeds=list(seeds))
        self.result.history.append(record)
        self.emit("attempt_started", dict(record))
        before = await self.assess(parent.policy, seeds)
        if before is None:
            record["status"] = "parent_failed"
            self.emit("rejected", dict(record))
            return
        examples = self.examples(parent.policy, seeds)
        self.result.reflection_calls += 1

        async def recorded(context, **kwargs):
            response, calls = await self.provider.acall(context, **kwargs)
            self.emit("raw_response", dict(prompt=context, response=response))
            return response, calls

        try:
            child = await self.reflect(
                parent.policy, examples, provider=SimpleNamespace(acall=recorded)
            )
        except (ValidationError, InvalidPolicy) as exc:
            record.update(status="invalid", error=str(exc))
        else:
            if child.source == parent.policy.source:
                record["status"] = "unchanged"
            else:
                after = await self.assess(child, seeds)
                record.update(before=fmean(before), after=fmean(after) if after else None)
                if after is None or fmean(after) <= fmean(before):
                    record["status"] = "not_improved"
                else:
                    self._mutation = record
                    return child
        self.emit("mutation", dict(record))
        return None

    def _accept(self, returns, *, seed_scores=None, error=None, feedback=""):
        if error is None:
            if seed_scores is None:
                if len(returns) != len(self.selection):
                    raise ValueError(
                        "Update requires the complete selection panel in configured seed order"
                    )
                seed_scores = dict(zip(self.selection, returns))
            if seed_scores.keys() != set(self.selection):
                raise ValueError("Evaluation returned an incomplete or unexpected seed panel")
            scores = tuple(seed_scores[seed] for seed in self.selection)
        policy = self._pending_policy
        if self._mutation is None:
            if error is not None:
                self.emit("rejected", dict(policy=policy.to_text(), error=error))
            else:
                self.result.population.append(Candidate(policy, scores))
        else:
            record = self._mutation
            if error is not None:
                record.update(status="selection_failed", error=error)
            else:
                self.result.population.append(Candidate(policy, scores, (record["parent"],)))
                record.update(
                    status="accepted",
                    candidate=len(self.result.population) - 1,
                    policy=policy.to_text(),
                    scores=list(scores),
                )
            self.emit("mutation", dict(record))
        self.emit(
            "measurement",
            dict(
                policy=policy.to_text(),
                seeds=list(self.selection),
                scores=dict(seed_scores or {}),
                failure=error,
                accepted=error is None,
            ),
        )

    def configure_panel(self):
        if not self.train or not self.selection or set(self.train) & set(self.selection):
            raise ValueError("Training and selection panels must be nonempty and disjoint")
        if any(type(s) is not int for s in (*self.train, *self.selection)):
            raise ValueError("Seed IDs must be integers")
        self.train = tuple(dict.fromkeys(self.train))
        self.selection = tuple(dict.fromkeys(self.selection))
        self._seed_panel = set(self.selection)
        if self.minibatch_size < 1:
            raise ValueError("Minibatch size must be positive")

    async def _proposals(self):
        self.configure_panel()
        if self.initial_policy is None:
            raise ValueError("GEPA requires an initial_policy")
        self.initial_policy.validate()
        if self.budget < len(self.selection):
            raise RuntimeError("Insufficient budget for complete panel")
        self.rng = random.Random(self.seed)
        self._mutation = None
        self.result.rollouts += len(self.selection)
        yield self.initial_policy
        if not self.result.population:
            return
        size = min(self.minibatch_size, len(self.train))
        while self.result.rollouts + 2 * size + len(self.selection) <= self.budget:
            weights = pareto_weights([c.scores for c in self.result.population])
            parent = self.rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
            child = await self.mutate(parent, self.rng.sample(self.train, size))
            if child is not None:
                self.result.rollouts += len(self.selection)
                yield child
        self.emit(
            "finished",
            dict(rollouts=self.result.rollouts, reflection_calls=self.result.reflection_calls),
        )

    async def run(
        self,
        initial_policy=None,
        *,
        train_seeds=None,
        selection_seeds=None,
        budget=None,
        minibatch_size=None,
        seed=None,
    ) -> Result:
        if initial_policy is not None:
            self.initial_policy = initial_policy
        if train_seeds is not None:
            self.train = tuple(train_seeds)
        if selection_seeds is not None:
            self.selection = tuple(selection_seeds)
        for key, value in dict(budget=budget, minibatch_size=minibatch_size, seed=seed).items():
            if value is not None:
                setattr(self, key, value)
        self.configure_panel()

        async def evaluate(policies):
            return await self.evaluate(policies, seeds=self.selection)

        await search(self, evaluate)
        return self.result
