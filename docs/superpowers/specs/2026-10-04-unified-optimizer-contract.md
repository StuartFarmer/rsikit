# Unified optimizer contract

Status: agreed architecture; implementation decisions specified below. No implementation in this change.

## Goal

All AlphaEvolve variants, ShinkaEvolve, EliteSearch, and LineageSearch work with one external propose → evaluate → update runner. The evaluator reports measurements. The optimizer owns proposals, interpretation, selection, repair decisions, and algorithm-specific completion. AlphaEvolve's `EvaluationResult` remains AlphaEvolve-owned and is constructed inside its update path.

## Public interface

Extend the existing core modules, without another optimizer base class or framework:

```python
# rsikit.evaluation; exported from rsikit
@dataclass(frozen=True)
class Measurement:
    scores: dict[int, float] = field(default_factory=dict)
    feedback: str = ""
    failure: str | None = None
    accepted: bool = True
    metrics: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)

# rsikit.optimization; exported from rsikit
class Optimizer(Protocol):
    @property
    def done(self) -> bool: ...
    @property
    def best(self) -> type[Policy] | None: ...
    async def propose(self) -> list[type[Policy]]: ...
    def update(self, results: Mapping[str, Measurement]) -> None: ...

async def search(
    optimizer: Optimizer,
    evaluate: Callable[[Sequence[type[Policy]]], Awaitable[Mapping[str, Measurement]]],
    *,
    on_checkpoint: Callable[[Optimizer], None] | None = None,
) -> type[Policy] | None: ...
```

Reuse `research.rewards.Measurement` by moving its definition into core, retaining a re-export at its former path. Preserve the first four fields and positional construction. Add only the measured metrics/descriptors already needed by paper AlphaEvolve. No universal `EvaluationResult`, opaque metadata bag, evaluator base class, `fit()`, or optimizer registry framework.

An evaluator object can supply its bound `evaluate` method; existing evaluator closures work unchanged. `rsikit.Evaluator` remains the existing single-episode collector. Raw `Episode` objects remain available for persistence and analysis, but are not the optimizer's feedback protocol.

## Measurement semantics

- `scores` preserves actual per-seed scalar measurements, not only an aggregate. Core validates integer seed keys, finite numeric values, nonempty string metric/feature keys, text feedback, boolean acceptance, and optional nonempty text failure. Booleans are not numeric scores. A failure forces `accepted=False`.
- An accepted result must contain scores or metrics. A rejected/failed result may contain partial or empty evidence. Evaluation integrations validate that accepted seed panels are complete; the optimizer validates the evidence its algorithm requires before changing state.
- All objectives use maximization, matching current algorithms; evaluators negate minimization objectives explicitly.
- `metrics` and `features` contain independently measured values. Optimizers derive aggregation, uncertainty, and archive decisions. No values are invented to fill missing evidence.
- Paper AlphaEvolve derives reward, worst reward, stability, mean-reward/reward-std descriptors from seed scores. Explicit measured metrics/features override corresponding derived defaults, preserving existing cascade precedence. Configured objective/descriptor requirements are checked before archive mutation.
- Screening stays in evaluator composition. Cheap rejection skips the expensive stage, returns `accepted=False` without `failure`, and does not trigger repair. External grading remains evaluation, never hidden execution inside synchronous `update()`.

## Round and lifecycle semantics

1. `propose()` chooses the next logical evaluation round; the runner does not supply `n`. Algorithm constructors/configuration determine round sizes and proposal limits.
2. At most one round is outstanding. Calling `propose()` with unresolved feedback raises `RuntimeError`. Concurrent calls on one optimizer are unsupported and rejected. A completed optimizer returns `[]` without generation.
3. Proposals are unique by policy ID. Multiple internal attempts for identical source/identity may share one measurement; the optimizer updates all associated attempts. Identity and ancestry rules remain unchanged.
4. `update()` receives exactly the outstanding IDs, including failures and rejections. Unknown/missing IDs, wrong types, mismatched seed panels, or repeated feedback raise before any selection/archive state changes. A protocol helper may validate common checks in `rsikit.optimization`; do not introduce a stateful base class.
5. Successful candidates are retained. Failures queue repairs; repairs run in the next `propose()`, under the existing shared generation/runtime repair allowance and model concurrency limit. Successful siblings are not resubmitted. Repaired policies retain attempt lineage even when their policy ID changes.
6. A repair evaluation round does not count as a new population generation or original proposal. Existing algorithm attempt budgets remain authoritative; provider calls, including repairs and planning, remain separately charged by `BudgetProvider`.
7. `update()` performs no asynchronous model calls or evaluation. Reflection/planning requiring I/O happens in `propose()` before it returns the next candidates.
8. `propose()` returns `[]` only when `done=True`. If all candidates fail generation, the optimizer advances internally within finite attempt/repair limits until it has candidates or is done. This prevents an external empty-batch spin loop.
9. `done` is algorithm completion, with no unresolved feedback or queued repairs. Runner cancellation, provider-budget exhaustion, and infrastructure failure are separate interrupted outcomes, not successful completion.
10. `search()` validates feedback, calls update, and returns `optimizer.best` (possibly `None`). It never checks algorithm types, calls private methods, or imports research code. It never takes ownership of evaluator/environment/provider resource cleanup; the experiment's context managers do.

Checkpoint callbacks run after proposals, after updates, and on exceptional exit. Callback failures are errors; if another exception is already propagating, log the secondary failure and preserve the original. Keep intra-proposal checkpoints where existing algorithms need to retain paid-call evidence. Adapters serialize algorithm records; the core runner knows no database schemas. Evaluators persist successful sibling evidence before returning or raising. Interrupted evaluations leave the optimizer's round pending for its supported recovery path.

## Scheduling and compatibility

The first shared runner uses complete rounds: generation finishes before evaluation starts; evaluation finishes before update. Model calls within proposal generation and episodes within evaluation still run concurrently with existing limits. Repairs settle the current logical generation before promotion or culling.

This intentionally replaces paper AlphaEvolve's streaming pipeline and EliteSearch's generation/evaluation overlap. It changes feedback timing, throughput, and potentially the search trajectory. Do not claim identical asynchronous trajectories or benchmark equivalence. Preserve selection rules and ancestry constraints; record the new `round-v1` optimization schedule in new run manifests. Streaming/partial updates are deferred, not silently emulated through private methods.

Keep thin compatibility entry points only where existing scripts depend on them. They delegate to core `search` and preserve documented return shapes where feasible. No second search implementation, polymorphic `update()` accepting both episodes and scores, or `fit()` loop. Public breaking changes (`propose(n)`, feedback format) are documented for this experimental package.

Preserve existing Elite unified resume, paper AlphaEvolve resume, exports, videos, holdout selection, and model-call accounting. New round state must be checkpointed where those recovery paths already exist. Do not promise new resume support for ShinkaEvolve, LineageSearch, or historical AlphaEvolve baselines. Legacy completed checkpoints can seed new rounds; an incomplete legacy streaming checkpoint must either be converted with a tested rule or be rejected before paid calls, with an actionable message. No guesswork about missing work or budgets.

## Algorithm boundaries

| Algorithm | Logical round and update boundary |
| --- | --- |
| AlphaEvolve original/improved | Configured batch of original attempts; preserve distinct variant founding rules and aggregate per-seed scores in update. |
| AlphaEvolve paper | Configured batch of attempts; measure founders before proposing descendants; build private `EvaluationResult`, then update the archive. |
| ShinkaEvolve | One configured generation plus any repair rounds; update populations/model gains and apply migration once per existing counting rules. |
| EliteSearch | Generate against a fixed previous-generation elite snapshot; promote exactly once after all new attempts and repairs terminate. |
| LineageSearch | Preserve exploration sweeps across active families, then global culling, then individual bonus expansions. Family updates await their full expansion and repairs; generation-only failures retain existing patience/attempt semantics. |

## Scope and success criteria

- Python 3.10+; existing dependencies only; existing unittest and Ruff tooling.
- Core never imports research/examples. Algorithms remain independent clients of core.
- The same runner executes all six implementations (three AlphaEvolve variants plus three other algorithms) using scripted providers and deterministic evaluators.
- New CLI built-ins cover `alphaevolve` (paper default, original/improved variants), `shinka`, `elite`, and `lineage`. Environment evaluation and seed selection remain shared. Existing custom-file `optimize(...)` remains compatible; these external files are not claimed to implement the protocol automatically.
- Algorithm-specific auxiliary choices (provider ensembles, embeddings, grading) stay in their Python APIs. Do not invent a general plugin/config system to expose every hook on the CLI.
- Paid calls are not needed for verification. Do not modify unrelated files. Removing global prompt configuration and broader library ergonomics are separate work.
