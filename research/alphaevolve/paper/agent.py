"""Published AlphaEvolve mechanisms with explicitly documented local archive rules."""

import json
from dataclasses import asdict, field
from statistics import fmean, pstdev
from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic.dataclasses import dataclass
from slick import parse, render

from rsikit.evaluation import episode_scores
from rsikit.policy import PolicyDefinition

from ..generation import Mutation, _PolicyResponse, apply_edits, evolution_regions
from ..improved.agent import AlphaEvolve as Baseline
from ..original.agent import Config as BaselineConfig
from ..original.agent import Guidance, PromptIdea
from .database import Candidate, Database
from .evaluation import NAMED_VALUES, EvaluationResult


def _saved_candidate(candidate):
    return {**asdict(candidate), "policy": candidate.policy.to_text()}


def _restored_candidate(saved):
    values = dict(saved)
    values["policy"] = PolicyDefinition.from_text(values["policy"])
    values["seed_scores"] = {int(k): v for k, v in values["seed_scores"].items()}
    return Candidate(**values)


@dataclass(frozen=True)
class Config(BaselineConfig):
    objective: str = "reward"
    features: dict[
        str,
        Annotated[
            tuple[float, float, int],
            BeforeValidator(lambda value: tuple(value) if isinstance(value, list) else value),
        ],
    ] = field(default_factory=dict)
    elite_fraction: float = Field(default=0.2, gt=0, le=1)
    exploration: float = Field(default=0.3, ge=0, le=1)
    reset_interval: int = Field(default=0, ge=0, le=0)
    migration_interval: int = Field(default=100, ge=0)
    migration_count: int = Field(default=1, ge=0)
    meta_interval: int = Field(default=25, ge=0)


class AlphaEvolve(Baseline):
    """Generate from persistent niche elites instead of one champion per island.

    Evaluation is supplied by the caller. All metrics maximize; `objective` selects
    the reported global best, while every metric has its own elite in each niche.
    """

    def __init__(self, task, provider, *, config=None, database_path=":memory:", **kwargs):
        config = config or Config()
        super().__init__(task, provider, config=config, **kwargs)
        self.database = Database(
            database_path,
            islands=config.islands,
            objective=config.objective,
            features=config.features,
            elite_fraction=config.elite_fraction,
            exploration=config.exploration,
        )
        state = self.database.load_state("optimizer")
        if state is not None:
            if state["task"] != task or state["context"] != self.context:
                self.database.close()
                raise ValueError("Population belongs to a different task/context")
            self.completed = state["completed"]
            self._attempt_offset = state["attempts"]
            self._prior_failures = state["failures"]
            self.generation_calls = state["generation_calls"]
            self.repair_calls = state["repair_calls"]
            self.meta_calls = state["meta_calls"]
            self.prompt_ideas = [PromptIdea(**idea) for idea in state["prompt_ideas"]]
            version, internal, gaussian = state["rng"]
            self.rng.setstate((version, tuple(internal), gaussian))
            round_state = state.get("round_state")
            if round_state is not None:
                self._attempt_offset = round_state["attempt_offset"]
                self._batch_number = round_state["batch_number"]
                self._seed_panel = (
                    set(round_state["seed_panel"])
                    if round_state["seed_panel"] is not None
                    else None
                )
                self._repairs = dict(round_state["repairs"])
                for saved in round_state["attempts"]:
                    row = dict(saved)
                    if row.get("policy") is not None:
                        row["policy"] = PolicyDefinition.from_text(row["policy"])
                    if row.get("parent") is not None:
                        row["parent"] = _restored_candidate(row["parent"])
                    row["inspirations"] = [
                        _restored_candidate(c) for c in row.get("inspirations", [])
                    ]
                    if row.get("idea") is not None:
                        row["idea"] = self.prompt_ideas[row["idea"]]
                    self.attempts.append(row)
                    if row.get("policy") is not None and row["status"] in (
                        "generated",
                        "repaired",
                        "execution_failed",
                        "evaluating",
                    ):
                        self._pending.setdefault(row["policy"].id, []).append(row)
                    elif row["status"] not in ("evaluated", "discarded"):
                        self._retry.append(row)
            elif state["attempts"] != state["completed"]:
                self.database.close()
                raise ValueError(
                    "Legacy streaming checkpoint has unresolved attempts; start a new run from an exported policy"
                )
        self._completed_offset = self.completed
        self._sync()
        self._log_leaderboard()
        self.checkpoint()

    def _sync(self):
        self.islands = self.database.champions
        self._best = self.database.best

    def checkpoint(self):
        attempts = []
        for record in self.attempts:
            row = dict(record)
            if row.get("policy") is not None:
                row["policy"] = row["policy"].to_text()
            if row.get("parent") is not None:
                row["parent"] = _saved_candidate(row["parent"])
            row["inspirations"] = [_saved_candidate(c) for c in row.get("inspirations", [])]
            if row.get("idea") is not None:
                row["idea"] = next(
                    i for i, idea in enumerate(self.prompt_ideas) if idea is row["idea"]
                )
            attempts.append(row)
        self.database.save_state(
            "optimizer",
            {
                "round_state": {
                    "attempts": attempts,
                    "attempt_offset": self._attempt_offset,
                    "batch_number": self._batch_number,
                    "repairs": self._repairs,
                    "seed_panel": sorted(self._seed_panel)
                    if self._seed_panel is not None
                    else None,
                },
                "task": self.task,
                "context": self.context,
                "completed": self.completed,
                "attempts": self._attempt_offset + len(self.attempts),
                "failures": (
                    self._prior_failures
                    + [
                        {key: row[key] for key in ("id", "raw", "error") if key in row}
                        for row in self.attempts
                        if row.get("error")
                    ]
                )[-3:],
                "generation_calls": self.generation_calls,
                "repair_calls": self.repair_calls,
                "meta_calls": self.meta_calls,
                "prompt_ideas": [asdict(idea) for idea in self.prompt_ideas],
                "rng": self.rng.getstate(),
            },
        )

    def close(self):
        try:
            self.checkpoint()
        finally:
            self.database.close()

    def sample(self):
        return self.database.sample(self.rng, inspirations=self.config.inspirations)

    def register_initial(self, policy, result: EvaluationResult, *, island=None):
        """Seed the archive with a caller-evaluated program (all islands by default)."""
        evolution_regions(policy.source)
        policy.validate()
        result = EvaluationResult(**vars(result))
        if not result.accepted:
            raise ValueError("Initial program must pass evaluation")
        candidate = self._candidate(policy, result)
        for target in range(self.config.islands) if island is None else (island,):
            self.database.register(candidate, target)
        self._sync()
        self._log_leaderboard()
        self.checkpoint()

    def _candidate(self, policy, result):
        return Candidate(
            policy=policy,
            score=result.metrics[self.config.objective],
            metrics=dict(result.metrics),
            features=dict(result.features),
            feedback=result.feedback,
            seed_scores=dict(result.seed_scores),
        )

    def discard(self, policy, reason):
        """Keep rejected evaluations in history but out of the breeding population."""
        for record in self._pending.pop(policy.id, []):
            record.update(status="discarded", error=reason)
            self._log_candidate(record)

    def update_results(self, results):
        # Validate the batch before consuming pending proposals.
        results = dict(results)
        schemas = set()
        for policy_id, result in results.items():
            records = self._pending[policy_id]
            if not isinstance(result, EvaluationResult):
                raise TypeError("Expected EvaluationResult")
            results[policy_id] = result = EvaluationResult(**vars(result))
            if result.accepted and self.config.objective not in result.metrics:
                raise ValueError(f"Missing objective metric: {self.config.objective}")
            if result.accepted:
                schemas.add(tuple(sorted(result.metrics)))
                for record in records:
                    parent = record["parent"]
                    self.database.validate(
                        self._candidate(record["policy"], result),
                        record["island"],
                        None if parent is None else parent.policy.id,
                    )
        if len(schemas) > 1:
            raise ValueError("Accepted results must share the same metric schema")
        for policy_id, result in results.items():
            records = self._pending[policy_id]
            if not result.accepted:
                self.discard(records[0]["policy"], result.feedback or "Evaluation rejected")
                continue
            for record in records:
                candidate = self._candidate(record["policy"], result)
                parent, idea = record["parent"], record["idea"]
                canonical = self.database.register(
                    candidate, record["island"], None if parent is None else parent.policy.id
                )
                if parent is not None:
                    scale = max(1.0, abs(parent.score))
                    idea.reward += max(0.0, canonical.score / scale - parent.score / scale)
                record.update(status="evaluated", score=candidate.score)
                self._log_candidate(record)
                self.completed += 1
                interval = self.config.migration_interval
                if interval and self.completed % interval == 0:
                    self.events.extend(
                        {**event, "completed": self.completed}
                        for event in self.database.migrate(
                            self.rng, count=self.config.migration_count
                        )
                    )
            self._pending.pop(policy_id)
        self._sync()
        self._log_leaderboard()
        self.checkpoint()

    def _evaluation_result(self, episodes):
        scores = episode_scores(episodes)
        values = list(scores.values())
        mean, std = fmean(values), pstdev(values)
        metrics = {"reward": mean, "worst_reward": min(values), "stability": -std}
        features = {"mean_reward": mean, "reward_std": std}
        for name, target in (("metrics", metrics), ("features", features)):
            measured = [
                NAMED_VALUES.validate_python(ep.infos[-1].get(name, {})) for ep in episodes.values()
            ]
            names = set().union(*(values.keys() for values in measured))
            if any(set(values) != names for values in measured):
                raise ValueError(f"All seeds must report the same {name}")
            target.update({key: fmean(values[key] for values in measured) for key in names})
        missing = self.config.features.keys() - features.keys()
        if missing:
            raise ValueError(f"Missing measured descriptors: {sorted(missing)}")
        return EvaluationResult(
            metrics=metrics,
            features={name: features[name] for name in self.config.features},
            seed_scores=scores,
        )

    def _accept_episodes(self, results):
        self.update_results({id: self._evaluation_result(r) for id, r in results.items()})

    def update(self, results):
        super().update(results)
        self.checkpoint()

    def update_scores(self, scores, *, seed_scores=None):
        """Scalar feedback for tasks without descriptor dimensions."""
        self.update_results(
            {
                policy_id: EvaluationResult(
                    metrics={self.config.objective: score},
                    seed_scores=dict((seed_scores or {}).get(policy_id, {})),
                )
                for policy_id, score in scores.items()
            }
        )

    async def mutate(
        self, parent, inspirations, guidance, failures, *, provider, record=None
    ) -> PolicyDefinition:
        schema = Mutation.model_json_schema()
        context = render(
            "paper/prompts/mutate.j2",
            instance=self,
            schema=schema,
            schema_json=json.dumps(schema, indent=2),
            parent=parent,
            inspirations=inspirations,
            guidance=guidance,
            failures=failures,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        mutation = parse(raw, Mutation)
        return PolicyDefinition.from_text(
            apply_edits(parent.policy.source, mutation.edits),
            name=mutation.name,
            description=mutation.description,
        )

    async def rewrite(
        self, parent, inspirations, guidance, failures, *, provider, record=None
    ) -> PolicyDefinition:
        schema = _PolicyResponse.model_json_schema()
        context = render(
            "paper/prompts/rewrite.j2",
            instance=self,
            schema=schema,
            schema_json=json.dumps(schema, indent=2),
            parent=parent,
            inspirations=inspirations,
            guidance=guidance,
            failures=failures,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["raw"] = raw
        return parse(raw, _PolicyResponse).to_policy()

    async def evolve_prompt(self, parent, ideas, failures, *, provider, record=None) -> str:
        schema = Guidance.model_json_schema()
        context = render(
            "paper/prompts/evolve_prompt.j2",
            instance=self,
            schema=schema,
            schema_json=json.dumps(schema, indent=2),
            parent=parent,
            ideas=ideas,
            failures=failures,
        )
        raw, _ = await provider.acall(context)
        if record is not None:
            record["meta_raw"] = raw
        generated = parse(raw, Guidance)
        return generated.instruction
