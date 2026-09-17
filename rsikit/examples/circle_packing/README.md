# Circle-packing optimizer integration example

Run an actual optimization experiment with a fixed problem, initial program,
evaluator, model prompts, repair loop, and measured selection.

## Run

From the RSIKit repository root, using the existing environment and adjacent
Slick checkout:

```sh
uv pip install --python .venv/bin/python -r rsikit/examples/circle_packing/requirements.txt
docker build -t rsikit-sandbox:local rsikit/sandbox
export OPENROUTER_API_KEY='your-key'
.venv/bin/python -B -m rsikit.examples.circle_packing --iterations 5
```

Start Docker Desktop before running. Build the image once (and rebuild when its
Dockerfile or worker changes). Each run starts one container and removes it on exit.

The default model is
[`openai/gpt-oss-120b:nitro`](https://openrouter.ai/openai/gpt-oss-120b:nitro),
through Slick's direct `OpenRouterAPI` provider. This makes paid requests using
your OpenRouter account. A shorter first run is `--iterations 2`.

## Compare strategies

Run the same problem, seed program, evaluator, and visualizations with:

```sh
.venv/bin/python -B -m rsikit.examples.circle_packing --strategy hillclimb --iterations 25
.venv/bin/python -B -m rsikit.examples.circle_packing --strategy alphaevolve --iterations 25
.venv/bin/python -B -m rsikit.examples.circle_packing --strategy eoh --iterations 25
.venv/bin/python -B -m rsikit.examples.circle_packing --strategy dgm-archive --iterations 25
.venv/bin/python -B -m rsikit.examples.circle_packing --strategy shinkaevolve --iterations 25
```

| Option | Behavior |
| --- | --- |
| `hillclimb` | Propose from the current best packing. |
| `alphaevolve` | Four islands, per-cell champions, exploration, cross-island inspirations, and weaker-half reseeding every 20 attempts. |
| `eoh` | Thought/source pairs using E1 diversity, E2 shared ideas, M1 structural change, M2 tuning, and M3 simplification. |
| `dgm-archive` | Retain feasible packings including regressions; sample by quality and child count; diagnose and modify the selected packing. |
| `shinkaevolve` | Weighted island parents, diff/full/crossover proposals, UCB model sampling, and periodic meta recommendations. |

`--iterations` counts individual candidate attempts, including rejections. It is
not an equal token/cost budget: DGM archive search makes two calls (diagnose +
modify) before repairs; HillClimb, AlphaEvolve, and EoH make one. For N attempts and R repairs, upper
bounds are `N*(2+R)` and `N*(1+R)` model calls. Malformed EoH JSON consumes an
attempt; source repair begins only after a source has been extracted successfully.

ShinkaEvolve automatically uses the modular template root. It permits at most
three proposals per attempt, each with up to R syntax repairs, plus one meta call
before each new generation following ten completed attempts. Without embeddings,
this bounds calls by `3*N*(1+R) + floor((N-1)/10)` for N > 0. The shared generation
deadline covers those calls. Geometry is evaluated once **after** novelty acceptance;
invalid geometry is recorded for future generations, without another repair loop.
The other strategies retain their existing feasibility-repair behavior.

The CLI defaults to one model and exact-duplicate rejection. Add
`--ensemble-model MODEL_ID` (repeatable) to include more mutation providers under
UCB selection. The base `--model` handles repair, novelty judging, and meta calls.
Programmatic callers can pass `shinka_providers=(provider_a, provider_b)` and
`shinka_options={"embed": async_embedding_callback}` to `experiment.run()` to
enable embedding-based novelty rejection and the LLM judge. Extra judge calls are
bounded by the proposal count. No embedding service or model is selected implicitly.
All these calls use the supplied API accounts; offline tests use scripted providers.

Shinka runs also save `shinka.json` with proposal/novelty decisions, migration and
meta events, the scratchpad, provider names, and Decimal model gains as strings.
`selections.json` includes the chosen model index and patch operation. See the
[RSIKit API](../../../README.md#shinkaevolve) for archive and normalization choices.

EoH defaults to a population of four: three additions beside the seed, then 20
attempts per full cycle (four per operator). Parents come from the same population
throughout that cycle, and elites are selected at its end. With successful
initialization, 23 attempts cover initialization plus a full cycle; initialization
rejections can delay it. For a smaller test, use `--population-size 2 --iterations 12`.
If stopped mid-cycle, the best evaluated solution is still returned and plotted.

`--seed` controls parent sampling, not model determinism. `--islands` changes the
AlphaEvolve or ShinkaEvolve island count. AlphaEvolve's diversity cell is `(number of radii > 0.15, number
of centers in the central half-square)`. DGM archive sampling normalizes scores
by `sqrt(10/pi)`, an area-based upper bound on the sum of radii.

AlphaEvolve proposals also receive the actual measured circles for the parent
and inspirations. Prompts prioritize better returned packings within the full
execution budget; numerical solvers are optional tools, and timeout repairs should
preserve useful geometry instead of resetting to the initial grid.

These are adaptations of the local search policies. AlphaEvolve is serial and
single-objective; EoH is seeded. **DGM archive search is not full DGM:** this
benchmark evolves an executable packing function, not the agent's own search
implementation. Diagnosis and modification use the fixed model directly.

## Modular prompts and reflection

Legacy prompts remain the default. Opt into the task-independent RSIKit operations:

```sh
.venv/bin/python -B -m rsikit.examples.circle_packing --prompt-mode modular --reflect --iterations 5
.venv/bin/python -B -m rsikit.examples.circle_packing --prompt-mode modular --instruction-file instruction.txt
```

These are new experimental prompts, not a wording-preserving migration. The
instruction file supplies plain text for `mutate` (HillClimb); programmatic callers
can pass an `instructions` mapping for any operation. With ShinkaEvolve, the file
applies to `diff`, `full`, and `cross`. All five strategies work in
modular mode. The CLI configures the checkout template root once; programmatic
callers must do the same for the generic proposer/reflection templates.

`--reflect` adds measured pair/failure guidance after candidate completion and
injects its bounded run-local memory into subsequent proposals. Ties need no
reflection call. A reflection failure preserves the completed candidate, saves
progress and propagates. The generation timeout also bounds its reflection.

Modular runs save `proposal_records.json` (rendered requests, raw responses,
parsed drafts and errors). Reflection runs additionally save `reflections.json`
and `reflection_attempts.json`. `summary.json` records prompt mode, instruction
text, reflection setting and model settings. Omitted provider settings such as
temperature are recorded as unknown, not inferred.

With reflection, add at most one model call per attempt to the call bounds above:
`N*(3+R)` for DGM and `N*(2+R)` for HillClimb, AlphaEvolve, and EoH.
For ShinkaEvolve without embeddings, the bound becomes
`3*N*(1+R) + floor((N-1)/10) + N` for N > 0. Iterations are still not an equal
cost allowance. The [offline prompt-search example](../prompt_search.py) shows an
application-owned shared call cap for generation, repair, reflection and revision.

## Problem and initial solution

Place exactly **10 circles in the unit square** `[0, 1] × [0, 1]`. Maximize
the **sum of their radii**. Radii can differ; all must be positive. Circles may
touch but cannot overlap or extend outside the square. Geometric comparisons use
a fixed tolerance of `1e-9`.

[initial.py](initial.py) returns a loose grid with radius `0.10` for every circle,
giving a baseline score of **1.0**. The model edits the function's implementation
to improve the packing produced by one call. Imports, loops, arithmetic, local
helper functions, NumPy, and SciPy optimization are allowed inside the marked block.

The program contract is:

```python
def pack_circles():
    # EVOLVE-BLOCK-START
    # Compute a packing here, optionally using numpy/scipy.
    return [
        # Exactly ten (x, y, radius) triples of ordinary Python numbers.
    ]
    # EVOLVE-BLOCK-END
```

The actual initial file contains all ten circles. This benchmark optimizes a
packing algorithm for this fixed ten-circle problem. Its single returned packing
determines fitness; it does not measure generalization across circle counts.
Convert NumPy arrays/scalars to ordinary lists/numbers before returning.

## What runs

1. Execute the initial function once in the sandbox; score its returned circles
   with the fixed host-side [evaluator](evaluate.py).
2. Ask the model for an improved program, supplying the incumbent and recent feedback.
3. Check protected text, execute the proposed function once, and validate its
   returned geometry. On failure, request a repaired version and evaluate it once.
   The default allows two repairs; `--max-repairs 0` disables them. Repeated identical
   versions within a repair chain reuse their recorded outcome.
   ShinkaEvolve instead repairs syntax, applies its novelty gate, and only then
   executes once; geometry failures remain invalid evaluated attempts.
4. Give the execution's score or invalid result to the strategy. Do not execute again.
5. Update selection state and repeat. Render the saved outputs at the end.

[system.j2](prompts/system.j2) contains the shared task instructions, included in
both [generate.j2](prompts/generate.j2) and [repair.j2](prompts/repair.j2).
[experiment.py](experiment.py) contains the complete composition and visible loop.

The sandbox uses one long-running Docker container with Python 3.12, NumPy 2.2.6,
and SciPy 1.15.3 preloaded. Each request forks a fresh child, loads the candidate,
and calls `pack_circles()` exactly once. The child has a ten-second wall-time budget
(`--evaluation-timeout`), cannot spawn processes/threads, and is killed on timeout.
Candidate prints are discarded so they cannot corrupt the result protocol.
Exceptions, invalid return data, and timeouts feed the repair loop for the other
strategies; ShinkaEvolve records them as invalid evaluation feedback. Worker
failures abort the run without retrying an uncertain execution.

The container has no network, host mounts, or API credentials; it runs as a non-root
user with a read-only root filesystem, a 16 MiB temporary filesystem, one CPU,
512 MiB memory, dropped capabilities, and no privilege escalation. This is a local
research sandbox, not a hostile multi-tenant execution service. The temporary
filesystem is shared across the run; candidate Python/module mutations are not.
The host computes fitness from returned numbers rather than trusting candidate
claims about scores. Source and returned JSON are each limited to 64 KiB.

Selection descriptors and plots use the recorded circles too. Neither executes
candidate code, so even a stochastic function is scored and plotted from the same
single result. Legacy literal-only runs can still be rendered without Docker.

## Results

The terminal reports each candidate's feasibility, score, repair count, and current
best score. Results go into a fresh `runs/circle-packing-*` directory:

Default names use UTC timestamps with microseconds, for example
`circle-packing-20260916T143005.123456Z`, so names sort chronologically.
Existing runs keep their names; `--output` still overrides the directory.

```text
initial.svg           Baseline packing
best.svg              Best packing, viewable in a browser
generations/0000.svg  Baseline packing above the score graph
generations/0001.svg  First generation, including score and selection status
progress.gif          Animated packing and accumulating score graph
best.py               Best program
summary.json          Baseline, best score, gain, status, model and run settings
selections.json       Operation, parent IDs, inspirations, and island per attempt
thoughts.json         EoH proposal descriptions, keyed by attempt ID
diagnoses.json        DGM archive diagnoses, keyed by attempt ID
history.json          Search attempts, lineage and evaluations
packings.json         Measured circles keyed by candidate ID; used by selection and plotting
repairs.json          Source and diagnostics for every completed repair run
candidates/0000/      Initial source, execution.json, circles.json, evaluation.json
candidates/0001/      First candidate that passed generation checks
checks/0000/          First executed version, returned data or error, and host evaluation
```

Each completed generation saves its source, scores, and diagnostics immediately.
All SVGs and the animation render once at completion or interruption, keeping
plotting out of the search loop and avoiding duplicate rendering. They use the full observed
score range, including the final best score. The graph's x/y limits stay fixed
throughout the animation. The large upper panel shows that generation's candidate;
the lower panel connects candidate scores and tracks the best score so far.
Rejected generations show the retained best packing with a rejection label and a
gap in the candidate-score line. Generation zero is the baseline. Titles include
generation, selection status, displayed score, and best score; circles are numbered.

The looping GIF shows one second per generation and holds the last frame for 2.5
seconds. Matplotlib and Pillow render it without a display server or FFmpeg.
To generate or rebuild these visuals for an existing run without model calls:

```sh
.venv/bin/python -B -m rsikit.examples.circle_packing.visualize runs/circle-packing-YOUR_RUN
```

Attempt IDs may have gaps when generation exhausts repairs. Their final source
and rejection still appear in history. Completed evidence and the best program
are saved on each iteration and on ordinary exceptions/cancellation; automatic
resume and crash-safe persistence are not implemented.

Exit status is **0** if a feasible packing improved on the initial score, **2** if
the run completed without improvement, and **1** for an unhandled execution error.
A model can fail to improve; inspect its proposals and feedback in the saved files.

Options: `--strategy hillclimb`, `--seed 0`, `--population-size 4`, `--islands 4`,
`--iterations 5`, `--max-repairs 2`, `--max-tokens 8192`, `--timeout 180`,
`--evaluation-timeout 10`,
`--model MODEL_ID`, and `--output NEW_DIRECTORY`. The timeout covers each complete
generation, including diagnosis and repairs. Defaults permit at most 15 model calls; transport
retries are disabled. Increase output tokens if reasoning leaves truncated code.

## Verify without model calls

```sh
RSIKIT_DOCKER_TESTS=1 .venv/bin/python -B -m unittest discover -s rsikit/tests
```

The integration tests use scripted responses and the real Docker worker: a failed
packing is repaired into a program using NumPy/SciPy that scores `1.25`, a regression
is rejected, and execution counts and saved plots are checked. Worker tests cover
fresh module state, errors, timeout recovery, filesystem/network restrictions, and
cleanup. Omit `RSIKIT_DOCKER_TESTS=1` to skip Docker-dependent tests. These tests
verify the integration, not live model quality.
