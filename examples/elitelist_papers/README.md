# EliteTable research programme

Working paper name: **EliteTable**. The implementation is currently named
`EliteSearch`; this folder reuses it without renaming or changing the algorithm.
Agreed direction: 2026-09-24. Background: [CONTEXT.md](CONTEXT.md).

**Paper 1 update, 2026-09-25:** six environments including CarRacing, ten independent
searches per environment, and **ten full generations even after target attainment**.
The current [protocol and reproduction checklist](PROTOCOL.md) and
[delivery plan/manuscript outline](outputs/.plans/elitetable-policy-search.md)
record the study scope. Preparation is now frozen in the standalone
[companion repository](companion/README.md): use its [launch handoff](companion/RUN_MAIN.md)
and [initial manuscript](companion/papers/elitetable-policy-search.md). The 60 main
searches will be run externally. Existing runs remain exploratory pilots.
Use the companion commands for main-v1; this development notebook is historical.

| Paper | Independently useful question | Scope |
| --- | --- | --- |
| 1. Evolving executable policies with an LLM | Can evaluated search improve policies over generations, and where does it stall? | Algorithm, held-out learning curves, interpretable policy changes, failures, resource use. |
| 2. What makes EliteTable work? | Which components contribute, and when? | Elite capacity, remixes, fresh proposals, feedback, repair, and selected interactions. |
| 3. Comparative performance | How competitive is it under explicit budgets? | Independent LLM sampling, single-policy refinement, evolutionary methods and appropriate RL baselines. |

An ablation paper needs explanatory findings, not only switches and scores. A
comparison paper must report resource trade-offs: matching LLM tokens does not
match RL environment steps or wall-clock cost. Do not split by individual
environment unless it yields a distinct scientific contribution.

## Paper 1

The working document is [paper1.ipynb](paper1.ipynb). It contains the paper outline,
protocol, editable experiment settings, launch commands, analysis and plots.
Running all its cells **does not launch simulations or call a model**.

**Claim to test:** EliteTable can produce policies that reach environment score
targets, or increase the best policy's score over generations. Report both target
attainment and improvement, including flat curves and unsuccessful runs.
The LLM weights are fixed;
policy source code evolves. These familiar tasks cannot establish that the LLM
invented previously unknown control strategies.

The notebook's per-run demonstration exports `figures/target-and-progress.csv`
and PDF/PNG plots. It reads completed generations from checkpoints, including
interrupted runs, and marks their status. Each row reports generation-one and
latest search scores, their difference, the recorded target and first generation
reaching it, plus available full-panel held-out confirmation for the same policy.
Undefined targets and missing evidence remain unknown. This descriptive evidence
does not require ablations or comparisons; those remain separate papers.

Describe the actual implementation: a frozen elite table breeds each generation;
new proposals, edits and multi-parent remixes are evaluated on a fixed search seed
panel; old elites and valid newcomers compete by mean reward. Repairs consume
additional calls. Generation one contains only new proposals. With the default
fractions, later generations allocate 20% new, 40% remixes and the remainder edits
(fractions are rounded down). Retention makes the best search score nondecreasing
by construction: that curve alone does not demonstrate generalization.

Equal mean rewards prefer **current-generation edits, then remixes, then existing
elites, then new proposals**. Existing elites use the incumbent category regardless
of how they were originally created. Within a category, the lower organism ID
wins. Higher scores always take priority. On resume, historical boards are kept
as recorded; future promotions use this rule. Treat resumes from the previous
age-only tie rule as a changed search protocol.

By default, the general CLI stops after a completed generation if its best elite's **mean search
reward** reaches Gymnasium's registered `reward_threshold` (CartPole: 475;
LunarLander: 200; BipedalWalker: 300). This is an operational search criterion,
not proof of held-out success. The current generation finishes; no further
generation is launched. `--generations` remains a maximum.

Use `--target-score VALUE` to override the target, or `--no-early-stop` to use the
full generation budget. Pendulum, Blackjack and both CliffWalking variants have
no registered threshold, so require an explicit target for early stopping. Custom
episode limits may require a different target. The resolved target is saved in
`experiment.json` under `config.target_score`; `summary.json` records
`reason: "target_reached"` and the actual generation count. Policies, curves and
held-out evaluations are still exported normally. Held-out results never drive
stopping. Existing running processes retain their original behavior.

Paper 1's main-study notebook sets `EARLY_STOP = False`, adding `--no-early-stop`
to every new launch command. New manifests also save `reporting_target` independently
of the stopping rule, so target lines and attainment counts remain available.
Legacy runs use their recorded `config.target_score`; no target is invented for
an old run that did not record one. Early-stopped pilots cannot become fixed-budget
main runs through resume: start new directories with the new protocol.

### Staged experiment

1. Plumbing pilot: CartPole, 3 independent searches, 5 generations, 20 candidates
   per generation, 5 elites, 10 search seeds and 100 held-out seeds.
2. Research pilot: add MountainCar, Pendulum, FrozenLake and BipedalWalker, keeping
   failures and flat curves. Use the pilot to choose a feasible fixed protocol.
3. Main experiment: 10 independent searches per environment, **10 full generations**,
   50 candidates per generation, 10 elites. Include CartPole, MountainCar, Acrobot,
   LunarLander, BipedalWalker and CarRacing. Freeze budgets,
   environment settings, prompts and seed panels before the main runs. Treat the
   pilots as exploratory; use fresh held-out seeds for the main experiment.

Use `openai/gpt-oss-120b:nitro` throughout this first study. Nitro routes by
throughput, not to a fixed provider; archive response metadata when available.
`--search-seed` controls search scheduling/parent sampling, not deterministic
hosted LLM output. Repeat whole searches; episodes and candidates within a search
are not independent search replicates.

Evaluate each generation's search-selected winner on held-out seeds **after
search finishes**, without repair, reselection or feedback. Repeated winners
reuse their cached held-out evaluations. Report uncertainty across whole searches,
final-minus-generation-one held-out improvement, failures and resource use.
For early-stopped runs, "final" means the stopping generation. Plot only observed
generations; later points can contain fewer runs and are conditioned on not
having stopped. Report those counts and the number reaching the search target.
Held-out curves need not be monotonic. A missing/failed evaluation is not zero
reward and must remain visible. Do not choose a winner using held-out scores.

Include a random-action reference and generation-one performance. Independent
sampling and broad comparisons are deferred to the follow-up study. Paper 1 makes
no claim that iteration adds value beyond budget-matched sampling.
That control is **not implemented here**: `--new-fraction 1` in the original
runner still exposes elite descriptions and scores, so is not independent sampling.

Stochastic games, particularly Blackjack, need more than 100 test episodes for
precise small effects; use pilot variance to set the final sample size. Default
CliffWalking is deterministic, so different seeds do not imply new situations.
FrozenLake's default map is fixed: new seeds test transition randomness, not map
generalization. Classic tasks can saturate in generation one; report that outcome.
Report BipedalWalker as "not solved within the tested budget/configuration", not
"cannot be learned". Do not claim superiority or universal improvement.

### Environment suite

Target: Gymnasium 1.3.0. Standard defaults unless explicitly noted below.

- Classic control: CartPole-v1, MountainCar-v0, MountainCarContinuous-v0,
  Acrobot-v1, Pendulum-v1.
- Toy text: Blackjack-v1, FrozenLake-v1, CliffWalking-v1, Taxi-v4.
- Box2D: LunarLander-v3, BipedalWalker-v3, CarRacing-v3.
- Additional variants: LunarLanderContinuous-v3, BipedalWalkerHardcore-v3,
  FrozenLake8x8-v1, CliffWalkingSlippery-v1.

Blackjack-v1 is Gymnasium Blackjack, **not** the repository's finite-shoe Blackjack.
LunarLander-v3 uses discrete actions with wind disabled; the continuous variant is
explicit. The older `examples.elitesearch` runner uses different LunarLander
settings. Do not pool its runs with these. CarRacing uses native RGB observations,
so interpret its results separately from state-vector control.
Blackjack and CliffWalking have no registered step cap: this runner explicitly
caps them at 500 steps. Every resolved limit and environment kwarg is recorded.

### Context supplied to the LLM

Every proposal, edit, remix and repair receives the environment's installed
Gymnasium documentation, resolved settings, observation/action spaces, step cap,
and evaluation contract. BipedalWalker also receives the complete observation
index mapping, normalization, motor ordering, torque semantics and reward details
checked against Gymnasium 1.3.0. The objective explicitly maximizes mean
undiscounted episode reward, including for negative-reward tasks. The actual
early-stop target (or disabled status), numeric episode/call timeouts and timeout
failure behavior are stated. Held-out results never guide search.

`context.txt` records the exact shared environment/evaluation context for each
attempt; `llm_calls.jsonl` records complete requests, including parent feedback
and repair diagnostics. The original context is also in `experiment.json`.
Resumes preserve the original environment description and rebuild the evaluation
contract with the attempt's limits. Resume contexts live under `resumes/`, with
their paths and SHA-256 hashes in `attempts.json`. The notebook checks context
history before pooling and excludes changed-context runs by default. Set
`INCLUDE_CHANGED_CONTEXTS = True` only for exploratory inspection.

## Run simulations separately

From the repository root, set the OpenRouter key (or use `.env`) and start Docker.
The launcher builds the application image automatically. Host Python is not used.

```sh
rtk proxy ./scripts/run examples.elitelist_papers.run --list-envs
rtk proxy ./scripts/run examples.elitelist_papers.run \
  --env CartPole-v1 --search-seed 0 \
  --output runs/pilot-cartpole-0
```

The second command makes paid API calls. Defaults are the small pilot above, with
up to 5 repairs per candidate, 4 model calls and 4 Docker episodes allowed concurrently. For a detached run with
all terminal output in a file, use the commands printed by the notebook (or):

```sh
rtk proxy mkdir -p examples/elitelist_papers/logs
rtk proxy sh -c 'nohup ./scripts/run examples.elitelist_papers.run --env CartPole-v1 --search-seed 0 --output runs/pilot-cartpole-0 >> examples/elitelist_papers/logs/pilot-cartpole-0.log 2>&1 < /dev/null & echo $!'
```

New run directories must be new. The CLI
prints its output path. `status.json` distinguishes running, completed and failed
work (a hard-killed process can leave status at running). The notebook reloads files
without contacting the model. The entire application, including generated policies, runs in Docker.

### Resume and execution timeouts

Stop the previous process before resuming; the run's file lock prevents two
processes from writing the same run. From the repository root:

```sh
rtk proxy ./scripts/run examples.elitelist_papers.run \
  --resume runs/pilot-Pendulum-v1-0 \
  --episode-timeout 10
```

`--episode-timeout` is wall-clock seconds per episode evaluation on each seed,
default **10**, for the worker to return its episode result. Docker startup,
queueing and final process cleanup are outside this limit; it is not a total across seeds.
The limit must be positive and finite. Lower limits apply to unfinished and future evaluations;
already completed scores remain cached and are **not** certified under the new limit.
The notebook prints a detached resume command so this terminal can remain free.
Existing runs inherit their saved limits unless explicitly overridden; use
`--episode-timeout 10` to apply the new limit when resuming an older run.

Resume restores the saved environment, model, seed panels, population, elite table,
parents, repair budgets and generation position. It completes an interrupted
generation before starting another. Completed candidates and episodes are reused.
An interrupted model call can need another request; hosted model responses are
not deterministic. An interrupted repair still consumes its recorded repair budget.
Candidates already discarded stay discarded.

You may override the episode timeout, worker/model concurrency, or increase `--generations`
(the **total** generation budget, not additional generations). Search-defining
settings such as the environment, seeds and model cannot change on resume.
Registered early stopping is preserved as originally configured, including for
older runs created before early stopping existed. A completed search can resume
unfinished held-out reporting; if its target was reached it will not breed more
generations. Further resumes inherit the last attempt's execution limits.

`experiment.json` and the original `source.zip` are retained. Each resume appends
to `attempts.json` and saves prior reports plus the current source snapshot under
`resumes/`. Updated reports are written at the run root. Changing timeouts creates
a mixed evaluation protocol; the notebook identifies these runs and excludes them
from pooled plots by default. Set `INCLUDE_CHANGED_TIMEOUTS = True` to inspect them
explicitly during exploration. Do not silently pool them with fixed-timeout runs.

### Output contract

- `experiment.json`: model, seed panels, search config, resolved environment,
  versions and revision information.
- `source.zip`: source/prompts used by the local runner and optimizer/evaluator,
  plus the lockfile, application Dockerfile and launcher for new runs. This is not yet a
  validated standalone release or a fully pinned worker build.
- `context.txt`: exact shared environment/evaluation context for the original attempt.
- `run.sqlite`, `exports/`, `best.py`: existing search evidence and policy code.
- `run.log`: existing generation log; detached stdout/stderr goes to `logs/`.
- `llm_calls.jsonl`: requests and returned response IDs, model/provider, usage,
  latency and call errors. Missing usage/cost is unknown, not zero.
- `summary.json`: existing search summary and final winner evaluation.
- `curves.csv`: one row per generation, including failed/missing winners, search
  and held-out mean, attempt/call/repair counts and held-out error.
- `heldout.json`: per-seed held-out results for generation winners and random policy.
- `status.json`: end-to-end state, elapsed time and errors, including post-search
  reporting failures. `summary.json` alone is not proof that the full run finished.
- `attempts.json`: original/resume attempts, execution limits, generation budget,
  source snapshot, status and elapsed time for each attempt.
- `resumes/`: prior report copies and code snapshots for resumed attempts.

The current evaluator persists episode returns, not environment-step counts.
Do not infer exact sample efficiency from candidate counts. API usage is recorded
only when returned by the service; reconcile unavailable billing against
OpenRouter before publication. The source snapshot does not freeze a container
image; retain the image digest recorded in experiment metadata when available.

## Notebook setup and checks

The analysis notebook can use any kernel with NumPy, Matplotlib and IPython;
its printed simulation commands still use the repository virtual environment.
To use that environment as the notebook kernel, install these if needed:

```sh
rtk proxy uv pip install --python .venv/bin/python ipykernel matplotlib
rtk proxy ./scripts/run unittest examples.elitelist_papers.test_paper1 examples.elitelist_papers.test_resume -v
```

The notebook uses the standard library and NumPy for analysis; pandas is not
required. It handles an empty runs folder and saves plots/tables under `figures/`.
Keep pilot and main results in separate experiment groups.

Primary references: [Gymnasium classic control](https://gymnasium.farama.org/environments/classic_control/),
[toy text](https://gymnasium.farama.org/environments/toy_text/),
[Box2D](https://gymnasium.farama.org/environments/box2d/),
[OpenRouter Nitro](https://openrouter.ai/docs/guides/routing/model-variants/nitro).
