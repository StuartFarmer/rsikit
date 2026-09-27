# EliteTable Hold'em

A population evaluator and runnable EliteTable search for six-player no-limit
Hold'em. The existing single-policy example and shared optimizer are unchanged.
PokerKit 0.7.6 supplies the rules and hand evaluation; the adapter supports side
pots, short all-ins and arbitrary legal integer bet sizes.

From the **repository root**, install the optional engine and build the worker:

```sh
rtk proxy uv pip install --python .venv/bin/python -r examples/elitelist_papers/poker/requirements.txt
rtk proxy docker build -t elitetable-poker:local \
  -f examples/elitelist_papers/poker/Dockerfile examples/elitelist_papers
```

The image extends the existing `rsikit-sandbox:local` image. No host source,
credentials or volumes are mounted into the evaluator container.

Run a small demo with 12 fixed reference policies and no API calls:

```sh
rtk proxy .venv/bin/python -B -m examples.elitelist_papers.poker demo \
  --output examples/elitelist_papers/poker/runs/demo \
  --rounds 2 --deals 50 --workers 4 --seed 123
```

Start an evolutionary search (requires `OPENROUTER_API_KEY` and makes paid calls):

```sh
rtk proxy .venv/bin/python -B -m examples.elitelist_papers.poker search \
  --output examples/elitelist_papers/poker/runs/search-0 \
  --population 48 --elites 12 --generations 5 \
  --rounds 4 --deals 100 --workers 4 --model-workers 4
```

Interactive terminals use the shared optimizer Rich style: nested generation
phases (generate and tournament), table blocks and elite capacity. The current
elite leaderboard stays pinned above the bars while logs scroll above it; scores
remain from the last completed generation until the next tournament finishes.
Small terminals show the top elites that fit. Held-out evaluation keeps the same
leaderboard visible; demo and benchmark evaluations show table bars. Logs also
append to `run.log`. Redirected output retains logs and the final summary rather
than terminal animation. Table logs include hand counts, elapsed time, hands/s,
the slowest policy's mean call time and compilation warm-up time. The host emits
an active-table heartbeat every five seconds while waiting. Policy failures log
the policy name/ID and actual exception immediately. Unrelated tables keep running;
failures are repaired together and unchanged completed blocks are reused.

Rebuild the worker after upgrading to the persistent evaluator service using the build
command above, then restart an existing search with `--resume`. Already-running
Python processes cannot pick up these host changes.

`rtk proxy` makes stdout look non-interactive. To retain live animation in your
terminal, insert `env TTY_COMPATIBLE=1 TTY_INTERACTIVE=1` after `rtk proxy`, e.g.:

```sh
rtk proxy env TTY_COMPATIBLE=1 TTY_INTERACTIVE=1 .venv/bin/python -B \
  -m examples.elitelist_papers.poker search \
  --output examples/elitelist_papers/poker/runs/search-0 --resume
```

Generation one proposes 48 policies. Subsequent generations propose 48 challengers
from 12 frozen elites, using the existing 20% new / 40% remix / remainder edit
allocation, rounded down. All 60 compete together; the best 12 current-field
scores become parents. Exact ties retain EliteSearch's edit/remix/incumbent/new
priority. Incumbents receive fresh scores. No score threshold stops self-play.

48 challengers plus 12 elites fills tables efficiently. Other population sizes
also work: schedules wrap cyclically until everyone has equal appearances. For
50 competitors, one round needs 25 tables and three appearances per player;
60 competitors need 10 tables and one appearance. Repairs that discard candidates
can therefore increase the remaining tournament's scheduling cost.

Each table plays `deals × table_size` hands, rotating every policy through every
seat on the same deals. Stacks reset to 200 chips (100 BB) per hand, blinds stay
1/2, and there is no rake. Tables regroup between rounds; a small randomized
schedule search reduces repeated opponent pairings. This is a fixed-stack league,
not an elimination tournament. Populations below six use smaller tables.

Fitness is net BB/100. With a population divisible by six, the example above
plays 2,400 hands per player per generation. Per-round scores, a descriptive
bootstrap interval, positional chip totals and action counts accompany rankings.
Intervals resample complete scheduling rounds, retaining duplicate correlations.
Few rounds give weak uncertainty estimates; this default is a plumbing/search
budget, not evidence that close policies are distinguishable. Positional totals
are raw chips, not independently normalized win rates.

The full observation and action contract is in [CONTRACT.md](CONTRACT.md).
`-1` represents a legal all-in raise, including a one-chip bet; `0` checks/folds;
`1` checks/calls; larger integers raise to that amount. Invalid actions and
policy deadlines fail the candidate. Failed challengers consume the existing
repair budget. Failed incumbents are removed without rewriting historical code.
Each generation generates policies, runs one scored self-play tournament, then
selects elites immediately. There is no separate reference-policy screening pass.
Tables collect failures in parallel and repairs use `--model-workers`. Queued
blocks containing a known failed policy wait for repair; unrelated tables finish.
Completed blocks are reused when the ordered policies, deals and configuration
are identical. Repairs rerun affected blocks; discards can change the schedule
and require more blocks. Rankings require a complete current tournament.
The block cache lasts for the current process and resets when the tournament
seed or configuration changes; resuming starts without cached blocks.
Infrastructure failures abort the run instead of inventing losses.

## Speed and isolation

**One persistent Docker container runs the entire CLI evaluation session.**
`--workers 4` means up to four concurrent table processes inside that container.
It is reused across rounds, repairs, generations and held-out evaluations;
benchmark measurements also share it. Startup logs identify the container and
shared forkserver. Standalone `Tournament` calls create one container for all
their blocks; Python callers can share `TablePool` explicitly across calls.

A clean, preloaded forkserver starts trusted table supervisors and private player
processes. Each player gets a distinct UID, temporary directory and policy RNG,
including across concurrent tables. Generated code never runs in the host,
forkserver, game engine or service. Player processes restart between duplicate
rotations; instances reset every hand. Module code and Numba kernels persist
within each player's rotation. Binary local IPC carries individual decisions.

The container's CPU limit scales with `--workers`. Its RAM limit defaults to
**32 GiB**, shared by all tables; use `--memory-gb 32` to set it explicitly or
override a saved limit on resume. This sets a cap, not a preallocation. Docker's
VM must have enough memory allocated to supply it. `--call-timeout` bounds
policy requests; trusted startup has a separate deadline. `--block-timeout`
bounds each block. A cancelled table's process group is killed and its temporary
files removed before its slot is reused; the container and other tables remain
available. The container is removed on run completion, error or interruption.
Cleanup waits are bounded, and Docker removal failures are reported and retryable.

`--policy-ms 5` sets a mean reset/act budget of 5 ms, measured in the supervisor
including IPC, with one second of cumulative grace per policy per block. Exceeding
it produces a repairable speed failure, including measured time and optimization
advice. `--warmup-timeout 15` separately bounds the first reset and first act in
each rotation; these calls are excluded from the mean budget so JIT compilation
is allowed. New budgets also apply when resuming runs created before these flags
existed. Existing generated policies are retained and repaired as needed.

Policies can use [Numba 0.64.0](https://numba.readthedocs.io/en/0.64.0/user/installing.html)
for expensive numeric strategies. Prompts recommend module-level
`@numba.njit(cache=False)` functions over numeric arrays and warming required
signatures in `reset`. Serial kernels work inside the sandbox; parallel kernels,
threads and disk caching are unavailable. Compiled kernels are recreated when
the player's isolated process restarts for the next duplicate rotation.

Measured on this workstation with 50 deals per table:

| Measurement | Result |
|---|---:|
| 300 check/call hands, repeated interpreter imports | 24.14 s |
| Same work, preloaded forkserver | 5.08 s |
| 1,200 mixed-policy table hands, one worker | 62.3 hands/s |
| Same mixed-policy work, four workers | 222.7 hands/s |

These are single timing samples, including Docker startup, with simple reference
policies. They are not LLM search speeds or guarantees for expensive policies.
The first comparison reduced startup cost 4.8×; the separate worker comparison
was 3.6× faster. [Measurement record](benchmark.json).

With live telemetry and Numba support enabled, the fixed-reference 1,200-hand
check measured 236.4 hands/s. The former fail-fast implementation returned a failing 48-policy field for
repair in 1.65 seconds, but discarded useful work. The current evaluator lets
unaffected tables finish and retains their results for retries. A slow saved
policy measured 35.8 ms/call and was stopped by the speed budget after seven
completed table hands. These are single samples, not speed guarantees.
[Feedback and speed measurements](feedback-benchmark.json).

The persistent service ran two consecutive 1,200-hand tournaments in the same
container: 6.13 s including initial startup, then 4.96 s with the service warm
(242 hands/s). These single samples used four workers and a 4 GiB cap, before
the default RAM cap was increased to 32 GiB.
[Persistent-container measurements](persistent-benchmark.json).

Reproduce the worker comparison with equal total work:

```sh
rtk proxy .venv/bin/python -B -m examples.elitelist_papers.poker benchmark \
  --output examples/elitelist_papers/poker/runs/benchmark \
  --rounds 2 --deals 50 --workers 4 --seed 123
```

## Evidence and resume

Search writes `checkpoint.json`, immutable `generation-NNN.json` reports,
`leaderboard.json`, `best.py`, `run.log`, full model request logs, status and experiment
configuration. Generation reports preserve both incumbent and challenger scores
and failed attempts; organism records retain sources and lineage.

After breeding finishes, historical winners play the same five fixed references
on fresh held-out deals. `heldout.json` and `curves.csv` show these separate scores;
they never affect selection or prompts. The references are deliberately simple,
so beating them does not establish strong poker play or low exploitability.

```sh
rtk proxy .venv/bin/python -B -m examples.elitelist_papers.poker search \
  --output examples/elitelist_papers/poker/runs/search-0 --resume --generations 10
```

Resume restores saved settings, sources, RNG state, parents, consumed repair
budget and historical results; the generation budget can increase and
`--memory-gb` can override the container RAM limit. A lock
prevents simultaneous writers. Interrupted tournaments restart as a whole; model
requests interrupted in flight may need another paid call. Completed held-out
measurements are reused. New output directories must not already exist.

Checks, from the repository root:

```sh
rtk proxy .venv/bin/python -B -m unittest examples.elitelist_papers.poker.test_poker examples.elitelist_papers.poker.test_display -v
rtk proxy env POKER_DOCKER_TESTS=1 .venv/bin/python -B -m unittest examples.elitelist_papers.poker.test_docker -v
```

Poker engine reference: [PokerKit game simulation](https://pokerkit.readthedocs.io/en/stable/simulation.html).
