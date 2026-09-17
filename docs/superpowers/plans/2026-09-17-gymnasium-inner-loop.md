# Gymnasium Inner Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Evaluate class-based policies against ordinary Gymnasium environments through one reusable, isolated episode loop.

**Architecture:** Task rules live in a trusted `gymnasium.Env`; generated `Solution` classes implement an asynchronous policy lifecycle. One runner records episodes, while a Docker-backed policy client keeps generated code outside the trusted environment. Start with one-step circle packing and unmodified multi-step CartPole.

**Tech Stack:** Python, Gymnasium, NumPy, stdlib dataclasses/asyncio/JSON, existing Docker execution infrastructure; retain Slick for future generation integration.

**Spec:** [Inner-loop design](../specs/2026-09-17-gymnasium-inner-loop-design.md).

## Global Constraints

- Python >=3.10; retain the existing Ruff conventions.
- Add Gymnasium >=1.3,<2 and an explicit NumPy dependency >=1.24,<3.
- No LLM weight training or fine-tuning.
- Implementation authorized after the planning turn; this plan is now implemented.
- No backwards-compatibility adapters for removed task functions or evaluator scripts.
- Do not weaken the existing isolation boundary to speed up the rewrite.
- Use small runnable smoke checks; comprehensive test hardening is a later phase.

The user authorizes breaking changes, deleted obsolete code, and broken old tests.
Do not spend this iteration preserving the old API or expanding a regression suite.
New smoke checks establish that the proposed contracts actually work. Execute
sequentially, reviewing each working milestone; no agent delegation is required.

## File map and migration decisions

| Path | Change / responsibility |
| --- | --- |
| `rsikit/policy.py` | New policy base, spaces, task instructions, and episode-local RNG. |
| `rsikit/episode.py` | New records, errors, and the only interaction loop. |
| `rsikit/envs/__init__.py` | Export custom environments without automatic registration. |
| `rsikit/envs/circle_packing.py` | Single trusted packing task class and geometric rules. |
| `rsikit/sandbox/__init__.py` | Replace function-call sandbox with `SandboxPolicy` and `run_program`. |
| `rsikit/sandbox/worker.py` | Rewrite as persistent class-instance worker. |
| `rsikit/sandbox/codec.py` | Explicit bounded JSON representation of supported spaces and values. |
| `rsikit/sandbox/Dockerfile` | Package worker, minimal policy contract, codec, and numeric dependencies. |
| `rsikit/examples/circle_packing/initial.py` | Replace function artifact with `Solution` class. |
| `rsikit/examples/circle_packing/__main__.py` | Replace generation CLI with an offline episode demo. |
| `rsikit/examples/cartpole.py` | New hand-written `Solution` and direct Gymnasium demo. |
| `rsikit/tests/test_inner_loop.py` | Small stdlib-unittest smoke suite for the new contract. |
| `rsikit/tests/test_sandbox_episode.py` | Small Docker integration checks for persistent state and cleanup. |
| `rsikit/__init__.py`, `pyproject.toml`, `README.md` | New public surface, dependencies, and launch instructions. |

Retire `LocalEvaluator` from `rsikit/evaluation.py`; retain `Evaluation` and
`EvaluationError` temporarily because existing search modules use them. The new
runner does not depend on this old score container. Preserve `selection.py`,
`archives.py`, `islands.py`, `population.py`, `strategies.py`, `shinka.py`,
`proposer.py`, `reflection.py`, and `prompt_search.py` as unintegrated research code;
remove their root re-exports instead of writing compatibility glue. Their future
integration is a separate plan.

Delete active obsolete packing paths `experiment.py` and `evaluate.py` after
moving the geometric rules into the environment. Replace obsolete sandbox tests
and packing CLI tests instead of patching them around removed signatures. Remove
old sine/prompt-search/concept demos from the primary README; those source files
can remain research examples until their respective later migrations. Reusable
packing visualization code can remain, but it is not part of this milestone.
Do not claim the entire historical suite is green; report the new suite separately.
Preserve attribution and `LICENSE.funsearch`/`NOTICE` when reusing packing content.

## Task 1: Policy lifecycle and trusted episode loop

**Files:** Create `rsikit/policy.py`, `rsikit/episode.py`, and
`rsikit/tests/test_inner_loop.py`; modify `pyproject.toml`.

**Interfaces produced:**

```python
policy = Policy(observation_space, action_space, instructions=task_instructions)
await policy.reset(seed=policy_seed)
action = await policy.act(observation)
await policy.close()

episode = await run_episode(
    make_env, make_policy,
    env_seed=1, policy_seed=2, max_steps=100, instructions=task_instructions,
)
```

The base reset installs `self.rng = numpy.random.default_rng(seed)` and seeds its
own copy of action_space for reproducible sampling; it must not mutate the trusted
environment's space object. Runner construction gives policy factories copied
space definitions. Subclasses must call the base reset when relying on the RNG.
Default `close` is an async no-op. Only `act` is abstract. Factories are synchronous;
remote startup belongs in async reset.

- [x] Add required keyword-only `instructions: str` to Policy construction and store it as `self.instructions`. Add optional `instructions: str | None = None` to both runner APIs. Explicit text takes precedence; otherwise read `env.unwrapped.instructions` before applying TimeLimit. Missing text is a caller configuration error; an explicit empty string is allowed. Do not require third-party environments to add an attribute or infer prose from docstrings.
- [x] Add dependencies and the policy lifecycle. Keep classes generic over observation/action types using Python-3.10-compatible typing.
- [x] Define `Transition` and `Episode` dataclasses with the exact fields in the spec. Keep `return_`/`length` derived. Use `EpisodeError` carrying `episode` for environment/infrastructure failures, and `PolicyError`/`PolicyTimeout` for remote candidate failures. Preserve candidate failure diagnostics without swallowing cancellation.
- [x] Implement `run_episode`: create env; resolve and record instructions; apply `TimeLimit`; construct policy with copied spaces and `instructions=resolved_instructions`; reset; validate observations; await actions; check space membership; step; validate finite reward and observation; copy records; stop on either ending flag; always close resources. Give the policy a copy of each observation so local mutation cannot corrupt environment state or the trace.
- [x] Put exception boundaries around policy calls separately from environment calls. Invalid actions never reach `env.step`. Host environment exceptions become `EpisodeError`; partial records remain attached. Candidate failures set episode status and preserve prior transitions. A failed constructor/reset produces a zero-transition failure record when attributable to candidate code.
- [x] Add one async smoke method using a two-step counter environment and a policy with a call counter. Reuse it to establish state persistence, exact reward summation, snapshotting, invalid-action rejection, and fresh instances on repeated calls. A separate one-step TimeLimit case establishes truncation semantics.
- [x] Extend that smoke check to capture instructions seen by the policy constructor. Confirm environment fallback, explicit override, deliberate empty text, and configuration failure when neither source exists. Confirm `info` is recorded but never passed to `act`; changing `self.instructions` cannot rewrite the Episode instructions field.

Core acceptance assertions in that smoke method:

```python
assert episode.status == "completed"
assert episode.length == 2
assert episode.return_ == 3.0
assert [t.action for t in episode.transitions] == [0, 1]
assert episode.transitions[-1].terminated
assert not episode.transitions[-1].truncated
assert limited.transitions[-1].truncated
assert invalid.status == "invalid_action" and invalid.length == 0
assert episode.instructions == captured_instructions == task_instructions
```

**Check:** `rtk proxy .venv/bin/python -B -m unittest rsikit.tests.test_inner_loop -v`.
Expected: the counter episodes complete; the invalid action is recorded without
stepping; repeating an episode starts the action counter at zero. Commit this
working unit before changing task examples.

## Task 2: Single-class circle-packing environment

**Files:** Create `rsikit/envs/__init__.py`, `rsikit/envs/circle_packing.py`;
rewrite `rsikit/examples/circle_packing/initial.py`; extend `test_inner_loop.py`.

**Consumes:** `Policy`, `run_episode`, `Episode` from Task 1.
**Produces:** `CirclePackingEnv(count=10)` and a baseline `Solution` policy.

- [x] Define canonical `CirclePackingEnv.instructions` alongside its task rules. Explain observation fields, action columns, circle constraints, objective, tolerance, infeasible reward, and one-step interaction. Keep the README consistent with this source; the runner reads the attribute rather than the README.
- [x] Move geometric validation from `examples/circle_packing/evaluate.py` into the environment. Use a Dict observation with `bounds` (four float64 values `[0, 0, 1, 1]`) and `count` (the configured integer, with a compatible Discrete space). Use the fixed-size float64 action Box and reward semantics specified in the design.
- [x] Implement `reset` using Gymnasium's base seeding and `step` as one submission. Sampled in-space actions must return an ordinary terminal result even when the circles overlap. Keep space errors distinct from geometric infeasibility.
- [x] Rewrite the baseline artifact to export `Solution(Policy)`; build ten radius-0.1 circles with x centers `0.1, 0.3, 0.5, 0.7, 0.9` and y centers `0.25, 0.75`. Return the float64 ndarray from async act.
- [x] Add a single packing smoke method that calls `check_env`, runs the baseline through the common runner, and then submits an overlapping but in-space action. The environment must reject the geometry through its task result, not an interface exception.

Acceptance assertions:

```python
check_env(CirclePackingEnv(), skip_render_check=True)
assert baseline.status == "completed" and baseline.length == 1
assert abs(baseline.return_ - 1.0) < 1e-9
assert baseline.transitions[-1].info["feasible"] is True
assert overlapping.status == "completed" and overlapping.return_ == 0.0
assert overlapping.transitions[-1].info["feasible"] is False
```

**Check:** rerun `rsikit.tests.test_inner_loop`. Commit the environment and baseline
as the first working task. No optimizer/provider/API call is involved.

## Task 3: Existing Gymnasium environment through the same API

**Files:** Create `rsikit/examples/cartpole.py`; extend `test_inner_loop.py`.

**Consumes:** unchanged `Policy` and `run_episode`.
**Produces:** runnable CartPole example with no RSIKit task adapter.

- [x] Define a `Solution` whose async act returns `int(observation[2] + 0.5 * observation[3] > 0)`. This is a plumbing baseline, not a claimed high-performing controller.
- [x] Pass a caller-owned instructions string explaining CartPole state coordinates, actions, objective, and standard reward/ending rules. Do not subclass, monkey-patch, or add an RSIKit task adapter to the Gymnasium environment. Confirm the episode records exactly the provided text.
- [x] Run `gym.make('CartPole-v1')` through the same factory-based runner with two fixed seeds and a 50-step cap. The CLI prints status, return, length, and final ending flags.
- [x] Add one smoke method confirming action membership, a nonempty trace, completion by termination or truncation, and reproducibility for this deterministic policy and seeded environment. Do not assert a particular performance score.

```python
assert episode.status == "completed"
assert 0 < episode.length <= 50
assert episode.transitions[-1].terminated or episode.transitions[-1].truncated
assert all(t.action in (0, 1) for t in episode.transitions)
assert episode.return_ == repeated.return_
```

**Check:** `rtk proxy .venv/bin/python -B -m rsikit.examples.cartpole` and the inner-loop
smoke suite. Commit this milestone. If the runner needs task-specific branches,
fix the interface before continuing; do not add a CartPole evaluator.

## Task 4: Execute a class safely while preserving episode memory

**Files:** Rewrite `rsikit/sandbox/__init__.py`, `worker.py`, and `Dockerfile`;
create `codec.py`, `rsikit/tests/test_sandbox_episode.py`.

**Consumes:** Task 1 lifecycle and episode runner, Tasks 2–3 task factories.
**Produces:** `SandboxPolicy` implementing the policy contract and
`run_program(program, make_env, *, env_seed, policy_seed, max_steps,
instructions=None, call_timeout=10.0) -> Episode`.

- [x] Replace fork-per-function execution with one container and one `Solution` instance per episode. Preserve the existing Docker restrictions. Install/copy only the worker's required contract and codec; do not import the host package's outer search modules inside the container.
- [x] Implement space/value encoding for Box, Discrete, Dict, Tuple, and Text as specified in the design. Add a small round-trip check for nested observations, ndarray dtypes, Text charset, and infinite Box bounds. Refuse unsupported spaces before executing source.
- [x] Give the host/client protocol sequential `start`, `act`, and `close` messages. Start carries source bytes, space definitions, resolved instructions, and seed. The worker imports source only inside Docker, constructs `Solution(observation_space, action_space, instructions=instructions)`, and awaits its lifecycle through one event loop. Act carries an observation and returns an action or a bounded candidate-error response. Keep user prints off the protocol stream.
- [x] Implement host deadlines and forced cleanup for startup/reset, actions, and close. Count initialization errors as candidate failures only after the infrastructure has reported readiness; malformed supervisor responses remain infrastructure failures. No automatic retries or repair inside evaluation.
- [x] Implement `run_program` by injecting a `SandboxPolicy` factory into `run_episode`; do not duplicate the rollout. The client owns source and timeout settings; its reset launches and initializes the remote instance, and its close destroys it.
- [x] Run one stateful counter policy through both paths and compare actions. Then run the packing artifact and CartPole artifact through Docker. Add one blocking-action case that proves a timeout kills the container. Docker checks skip explicitly when Docker/image is unavailable; never fall back to host execution of generated code.
- [x] Make the counter policy's first action depend on the received instructions so the local/isolated comparison detects lost or modified text. Include newline and non-ASCII text in the start-message round-trip check; task instructions remain data and are never interpolated into Python source.

Acceptance assertions:

```python
assert [t.action for t in isolated.transitions] == [0, 1]
assert isolated.return_ == local.return_
assert timed_out.status == "timeout"
assert packing.status == "completed" and abs(packing.return_ - 1.0) < 1e-9
assert cartpole.status == "completed" and cartpole.length > 0
```

**Checks:**

```sh
rtk proxy docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
rtk proxy .venv/bin/python -B -m unittest rsikit.tests.test_sandbox_episode -v
```

Commit the working isolated vertical slice. This is the main execution change;
do not sacrifice isolation or memory correctness to keep the task small.

## Task 5: Replace the active API and retire the old evaluation path

**Files:** Modify `rsikit/__init__.py`, `rsikit/evaluation.py`, `README.md`,
`rsikit/PLAN.md`, packing `__main__.py`/`README.md`; delete packing `experiment.py`
and `evaluate.py`; replace obsolete sandbox/packing command tests.

- [x] Export only the new inner-loop surface at the root: `Policy`, `Episode`, `Transition`, `EpisodeError`, `run_episode`, and `run_program`. Leave custom task imports under `rsikit.envs` and backend details under `rsikit.sandbox`.
- [x] Remove `LocalEvaluator` and the retired function-name dispatch/result-file plumbing. Preserve old score models solely for unintegrated research modules. Search callers before deleting and remove active references rather than adding compatibility aliases.
- [x] Replace the packing CLI with a baseline program episode and an optional user-supplied program path. Generated program paths always use `run_program`; trusted local classes are an explicit development path. Both demos print episode evidence rather than a generation/history summary.
- [x] Rewrite the README around one environment, one Solution policy class, and one episode. Document async reset/act/close, spaces versus semantic instructions, environment fallback and caller override, stable task instructions versus evolvable policy prompts, seeds, record fields, trusted vs isolated execution, supported remote spaces, and the deferred model API gateway. Reserve `model` for the trained model/API service and keep `organism` as optional evolutionary vocabulary. Mark old plan/docs as historical rather than current setup instructions.
- [x] Keep one explicit new-check command and distinguish its results from legacy suite failures. Run Ruff only on touched code, then verify package construction and imports from outside the checkout with the new example files/assets included.

Final acceptance commands:

```sh
rtk proxy .venv/bin/python -B -m unittest rsikit.tests.test_inner_loop rsikit.tests.test_sandbox_episode -v
rtk proxy .venv/bin/python -B -m rsikit.examples.circle_packing
rtk proxy .venv/bin/python -B -m rsikit.examples.cartpole
rtk proxy uv build
```

Do not require a green legacy suite, benchmark improvement, model calls, mutation,
or archive integration to finish this plan. Record Docker skips and any deferred
packaging/runtime issue honestly rather than claiming full validation.

## Completion boundary

The work is complete when a Solution class can be exchanged for another without
changing either task or runner; circle packing and native CartPole use the same
episode loop; class memory survives steps but not episodes; invalid output,
ordinary failure, truncation, and runtime failure stay distinguishable; and
generated code cannot execute in the trusted evaluator process. Effective task
instructions reach local and isolated policies unchanged and are recorded on
episodes; native Gymnasium environments need no RSIKit-specific modification.

The next API discussion can then use real episode records and actual policy
artifacts to design the adaptive agent's creation/evaluation boundary. Nothing in
this plan commits that later layer to a particular optimizer or representation
for mutation.

## Execution record — 2026-09-17

All five milestones implemented. The new smoke suite has eight tests, including
real Docker episodes; all pass without skips. Circle packing scores 1.0 and
CartPole completes the 50-step demonstration cap. Wheel/source-archive build and
installed-wheel smoke checks pass. Lint/format cover touched Python code.

Implementation decisions:
- The worker forks once per episode to separate the candidate's output from its
  trusted protocol supervisor; it never forks per action.
- Docker builds from the repository root and exposes Policy at both
  `rsikit.Policy` and `rsikit.policy.Policy` inside the minimal worker package.
- Episode status starts as `incomplete` for partial evidence attached to evaluator
  failures; cleanup diagnostics are retained separately from the primary error.
- Removed the retired generation-history visualizer along with obsolete packing
  orchestration/tests; the earlier source remains in Git history.
- Work was assembled in a temporary isolated checkout and delivered as one
  coherent rewrite rather than creating intermediate milestone commits.
- Historical outer-loop tests and examples are not migrated or counted as the
  active suite. No compatibility promises or model calls were added.
