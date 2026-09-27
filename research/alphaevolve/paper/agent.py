"""Published AlphaEvolve mechanisms with explicitly documented local archive rules."""

from dataclasses import asdict, dataclass, field

from slick import prompt

from rsikit.generation.edits import Mutation, Program, check_program

from ..improved.agent import AlphaEvolve as Baseline
from ..original.agent import Config as BaselineConfig
from ..original.agent import Guidance, PromptIdea
from .database import Candidate, Database
from .evaluation import EvaluationResult


@dataclass(frozen=True)
class Config(BaselineConfig):
    objective: str = "reward"
    features: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    elite_fraction: float = 0.2
    exploration: float = 0.3
    reset_interval: int = 0
    migration_interval: int = 100
    migration_count: int = 1
    meta_interval: int = 25

    def __post_init__(self):
        if self.reset_interval:
            raise ValueError("The paper variant uses migration_interval, not champion resets")
        for value in (
            self.migration_interval,
            self.migration_count,
            self.meta_interval,
            self.inspirations,
            self.max_repairs,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Intervals and counts must be nonnegative integers")
        if self.mode not in ("diff", "rewrite"):
            raise ValueError("Mode must be diff or rewrite")


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
        self._sync()
        self.checkpoint()

    def _sync(self):
        self.islands = self.database.champions
        self._best = self.database.best

    def checkpoint(self):
        self.database.save_state(
            "optimizer",
            {
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
        self.checkpoint()
        self.database.close()

    def sample(self):
        return self.database.sample(self.rng, inspirations=self.config.inspirations)

    def register_initial(self, policy, result: EvaluationResult, *, island=None):
        """Seed the archive with a caller-evaluated program (all islands by default)."""
        check_program(policy._implementation)
        if not result.accepted:
            raise ValueError("Initial program must pass evaluation")
        candidate = self._candidate(policy, result)
        for target in range(self.config.islands) if island is None else (island,):
            self.database.register(candidate, target)
        self._sync()
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

    def update_results(self, results):
        # Validate the batch before consuming pending proposals.
        schemas = set()
        for policy_id, result in results.items():
            records = self._pending[policy_id]
            if not isinstance(result, EvaluationResult):
                raise TypeError("Expected EvaluationResult")
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
        self.checkpoint()

    def update(self, scores, *, seed_scores=None):
        """Scalar compatibility for tasks that configure no descriptor dimensions."""
        self.update_results(
            {
                policy_id: EvaluationResult(
                    metrics={self.config.objective: score},
                    seed_scores=dict((seed_scores or {}).get(policy_id, {})),
                )
                for policy_id, score in scores.items()
            }
        )

    @prompt(template="paper/prompts/mutate.j2", output_type=Mutation)
    async def mutate(self, parent, inspirations, guidance, failures, *, generated: Mutation):
        return generated

    @prompt(template="paper/prompts/rewrite.j2", output_type=Program)
    async def rewrite(self, parent, inspirations, guidance, failures, *, generated: Program):
        return generated

    @prompt(template="paper/prompts/evolve_prompt.j2", output_type=Guidance)
    async def evolve_prompt(self, parent, ideas, failures, *, generated: Guidance):
        return generated.instruction
