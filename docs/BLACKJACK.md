# Finite-shoe blackjack

`rsikit.envs.BlackjackEnv` is a Gymnasium environment for evolving playing and
betting policies. It uses the existing NumPy/Gymnasium dependencies, with no
compiler, GPU, or additional packages. One episode defaults to 24 shoes, with
many rounds per shoe. Ten seeds therefore cover roughly 10,000 table rounds.

```python
from rsikit.envs import BlackjackEnv

env = BlackjackEnv(decks=6, penetration=0.75, shoes_per_episode=24)
observation, info = env.reset(seed=42)
# Betting phase: choose the index of a wager before seeing this round's cards.
observation, reward, terminated, truncated, info = env.step(0)
```

Use the normal Gymnasium loop and select only actions enabled by
`observation[42:]`. `reset()` starts a new multi-shoe episode; **do not reset
between rounds or shoes**. `info['round_complete']` marks a finished round, even
if its net reward is zero. Natural blackjacks can finish on the betting step.
The environment automatically shuffles between shoes after settlement, never
during a hand, and terminates only after the final shoe.

For this repository's trusted policy runner:

```python
from copy import deepcopy
from examples.blackjack import Solution
from rsikit import Evaluator

# Inside an async function:
with BlackjackEnv() as env:
    policy = Solution(deepcopy(env.observation_space), deepcopy(env.action_space))
    try:
        observation, info = env.reset(seed=42)
        await policy.reset(seed=1)
        episode = await Evaluator(env, policy).run(observation, info=info)
    finally:
        await policy.close()
print(episode.total_reward, len(episode))
```

`Solution` is an intentionally weak starting policy, not an optimal strategy.
The environment's `instructions` describes the rules and observation layout to
generated policies. Its `Box`/`Discrete` spaces work with the existing sandbox
codec. Only the observation reaches the policy through `act()`.

The same example source can run through `run_program(Path("examples/blackjack.py"),
BlackjackEnv, env_seed=42, policy_seed=1)` with the existing Docker policy worker.
When using `Executor`, which also moves the environment into Docker, rebuild
the worker image after adding this environment so the new module is installed.

## Actions and rules

During betting, action `i` chooses `bet_sizes[i]`. During play, `0` stands, `1`
hits, `2` doubles, and `3` splits. Invalid or masked actions raise Gymnasium's
`InvalidAction` before changing the game. The episode runner converts this to a
recoverable `PolicyError`, allowing optimizer repair without interrupting sibling
or failing sibling candidates. The action space is `Discrete(max(4, len(bet_sizes)))`.

A zero wager **sits out one round** (action `4` with the default bets). The agent
observes a stand-in player hitting below 17 and standing otherwise, without
splitting or doubling, followed by normal dealer play. This advances the shoe in
one step, returns zero reward, and goes back to betting unless the episode ends.
Only publicly exposed cards are returned; a dealer hole stays hidden if the
stand-in player busts. `info['sat_out']` marks these completed rounds. The agent
cannot sit out after placing a positive wager.

Defaults are six decks, stand on soft 17, 3:2 natural blackjack, double on any
initial two cards including after splitting, and at most four player hands.
Equal card values can split, including ten-valued cards. Split aces get one
additional card each and cannot be resplit. A split 21 pays 1:1. Dealer blackjack
is checked before player decisions; two naturals push. There is no insurance,
surrender, burn card, or bankroll limit. The stand-in player appears only during
sit-outs. Sitting out every round is a valid zero-profit baseline.

Reward is total net profit across the round's player hands, in base units;
intermediate actions earn zero. Doubles and splits require additional stakes.
With an eight-unit bet, a natural earns 12 units, an ordinary win earns 8, and
a doubled loss costs 16. `info` contains only round-completion statistics.

Constructor settings:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `decks` | `6` | Integer from 1 to 8; exact 52-card decks, without replacement |
| `penetration` | `0.75` | Requested cut-card fraction, in `(0, 1]` |
| `bet_sizes` | `(1, 2, 4, 8, 0)` | Integer stakes from 0 to 10,000; zero sits out |
| `hit_soft_17` | `False` | Dealer hits rather than stands on soft 17 |
| `max_split_hands` | `4` | Integer from 1 to 4; 1 disables splitting |
| `double_after_split` | `True` | Allow doubling a split hand's first two cards |
| `shoes_per_episode` | `24` | Positive integer; use `1` for historical single-shoe evaluation |

## Observation and memory

The observation is a fresh `int32` array, length 47 with default bets. Previously
returned arrays do not change when the environment steps. Card values are 1 for
aces, 2–9 for number cards, and 10 for tens and faces. Zero is padding.

| Index | Public information |
| --- | --- |
| `0` | Phase: 0 betting, 1 playing |
| `1` | Active player hand's total |
| `2` | During play: usable ace. During betting: fresh-shoe flag in multi-shoe episodes |
| `3` | Dealer upcard |
| `4` | Number of cards in the active hand |
| `5` | Pair value, or zero unless two cards have equal values |
| `6` | Active hand index, starting at zero |
| `7` | Number of player hands in the round |
| `8` | Active hand's wager |
| `9` | Number of newly exposed cards in this observation |
| `10:42` | Newly exposed cards, followed by zero padding |
| `42:` | Legal-action mask, one entry per action |

Fields 1–8 are zero during betting except the fresh-shoe flag at index 2.
A terminal observation has an all-zero action
mask. There is **no running count, true count, remaining-rank histogram, shoe
order, hidden-card value, or precomputed strategy** in either observation or
`info`. The deck count and rules are public configuration, supplied in instructions.

Read `obs[10:10 + obs[9]]` once per observation, including betting observations:
the settlement step reveals the dealer's hole card and any dealer draws. The
upcard and player cards are emitted when dealt, and never emitted twice. Moving
an existing card into a split hand is not a new exposure. If every player hand
busts, the dealer hole card stays unseen and the dealer does not draw.

A policy must maintain its own memory to learn from exposures. In multi-shoe
episodes, `obs[0] == 0 and obs[2] == 1` announces a fresh shoe before the next
wager, including the initial reset. At a shuffle boundary, process the exposed
cards from the completed round first, then clear shoe-specific counting memory.
The signal contains no count or hidden composition. `info['shoe_shuffled']`
reports the same boundary to the runner. `policy.reset()` occurs once per seed,
so the policy handles these boundaries within `act()`. Single-shoe mode retains
the original observation values. A feedforward policy cannot remember earlier rounds.

For a two-stage evolution curriculum, begin with `bet_sizes=(1, 1, 1, 1, 1)` to
optimize playing decisions without sitting out, then use `(1, 2, 4, 8, 0)` to evolve
pre-deal betting, sitting out, and memory. Both stages use identical observation
shapes. Evaluate multiple held-out seeds
and deck sizes; a good result on a few shoes is dominated by payoff variance.
The environment enables this curriculum; it does not train an agent itself.

## Test with EliteSearch

For a new test run with automatic leader videos:

```sh
./scripts/run examples.blackjack_train --output runs/blackjack-test
```

This uses 3 generations, 8 candidates per generation, 4 retained elites, 10
training seeds, 24 shoes per seed, and 10 separate held-out seeds. It requires
`OPENROUTER_API_KEY` and the local Docker image, and makes paid model calls.
All existing EliteSearch options can override these defaults.

Each seed has a 60-second wall-clock limit, adjustable with `--episode-timeout`.
Exceeding it produces a recoverable policy failure for repair or discard; other
episodes keep running. This is separate from the 10-second per-action timeout.
Video replays use the same episode limit saved in `experiment.json`.

Once a generation finishes scoring, its top four leaders are queued for video
export. Rendering runs alongside subsequent training, with one export at a time
to keep output consistent. `runs/blackjack-test/videos/index.html` updates after
each generation's videos finish. Each leader video includes all ten training
seeds at **30 fps, one action per frame**. The command waits for queued videos
before exiting. Repeated leaders reuse finished videos. `--video-top` and
`--video-workers` control the number of leaders and rendering processes; each
generation's export log is saved under `videos/generation-NN.log`. The same
feature is available on `examples.elitesearch --env Blackjack --video-top 4`.

The existing optimizer CLIs accept `--env Blackjack`, using the default six-deck
rules and allowing sitting out. Start with a small EliteSearch run:

```sh
# Rebuild so Docker can import the new environment module.

# Requires OPENROUTER_API_KEY; this makes paid model calls.
./scripts/run examples.elitesearch \
  --env Blackjack \
  --elites 3 --population 8 --generations 3 \
  --generation-concurrency 4 --concurrency 4 \
  --shoes-per-seed 24 --seeds {0..9} --heldout-seeds {1000..1009} \
  --output runs/blackjack-long
```

Brace expansion works in zsh/bash. Each seed evaluates 24 shoes, keeping policy
memory across hands and signaling each reshuffle. Ten seeds cover 240 shoes,
roughly 10,350 rounds with the baseline policy. Sitting-out rounds count toward
this total; split hands are not counted separately. EliteSearch defaults to ten
search seeds and 24 shoes per seed. Leave `--max-steps` unset so hands are settled
before an episode ends. The objective is average net profit per 24-shoe episode;
its scale differs from historical single-shoe scores. EliteSearch keeps
the top policies and breeds edits/remixes for the next generation; the final
winner is evaluated on the separate held-out seed panel.

Inspect `runs/blackjack-long/best.py`, `leaderboard.json`, `summary.json`, and
`run.log`. The command attempts 24 candidates plus any repair calls. It tests the
optimization pipeline. Compare against the zero-profit always-sit-out baseline
and use a larger disjoint panel when assessing a counting advantage. Do not use
held-out results to guide the search.

The shared environment factory also enables `--env Blackjack` for the existing
AlphaEvolve, ShinkaEvolve, and LineageSearch CLIs. Start a new run when changing
episode length; historical scores are not comparable to the new episode totals.

## Table preview

The optional renderer produces a canonical SVG frame with the table, individual
hands and wagers, last action, and cumulative net points per action. `render_svg()`
returns a standalone SVG with bundled Latin Modern fonts; `render()` rasterizes
that SVG through resvg to a 1280×720 RGB frame for Gymnasium/video. Each frame is one action;
playback is 30 frames and 30 actions per second (`render_fps=30`), with no repeated
or intermediate animation frames.
Generate the initial design preview with:

```sh
./scripts/run examples.blackjack_screen --output runs/blackjack-screen.png
./scripts/run examples.blackjack_screen --output runs/blackjack-screen.svg
```

This saves `runs/blackjack-screen.png` on the host from a real baseline
replay: seed 301, action 68, two split hands, and +3 net points. Use `--seed`,
`--steps`, and `--output` to select another frame. Install the `video` extra
for SVG-to-video rendering; no system fonts or TeX installation are needed.

`BlackjackRenderer(BlackjackEnv())` wraps only visual runs; ordinary training
keeps the original environment. Its private card identities preserve the exact
shuffle and do not enter observations. The dealer's hole stays hidden until
revealed by the game. The chart tracks settled reward, so wagers and hits leave
it flat until the round finishes. Cards, suits, labels, and charts are vector
geometry/text; unrevealed card identities are absent from the SVG as well as the video.

Saved policy exports include a final `.svg` snapshot and a `.svg.zip` archive
alongside each MP4. The archive contains `frames/00000000.svg` onward, shared
`fonts/`, and `timeline.json` with 30 fps and seed segment boundaries. Extract the
whole archive to preserve relative font paths. Frame zero is the first post-action
state. Frames retain stable scene IDs for future web animation; no browser player
or animation runtime is required during training. Theme-version changes rerender
media while retaining verified action traces.

## All-seed generation videos

Render the best saved elite from every completed generation of the current run:

```sh
./scripts/blackjack-videos
```

This defaults to `runs/blackjack-parallel-20261003-225001` and writes MP4s under
`videos/generation-NN/`, with a gallery at `videos/index.html`. It uses all saved
training seeds and 24 shoes per seed. Incomplete generations are skipped; rerun
the script to pick up newly completed generations. Docker is required, and the
existing launcher rebuilds the image to include renderer fixes.
The script uses host Python 3 to take a consistent SQLite backup before launching
Docker, avoiding live database reads across Docker Desktop's file share. The
temporary snapshot is removed when the exporter exits; the source is unchanged.

Pass another run (relative to the repository root) and optional exporter flags:

```sh
./scripts/blackjack-videos runs/blackjack-parallel-20261003-225001 --seeds 0 --workers 2
```

For the top four elites per generation instead:

```sh
./scripts/run examples.blackjack_videos runs/blackjack-smoke3
```

This creates `runs/blackjack-smoke3-all-seeds-videos/index.html`, with the top four
saved elites per completed generation. Each video concatenates **every search
seed from experiment.json**, in order, at 1280×720 and 30 fps. Each action supplies
exactly one frame, with no extra reset frames. Seed and shoe labels identify the
current episode; the points timeline accumulates across all seeds. Policy memory
resets at seed boundaries. Original scores are checked against every replay.

The exporter reads the saved `shoes_per_seed` setting; runs predating this option
retain one shoe per seed. It runs generated code in the application container,
saves action traces, and streams frames to FFmpeg instead of retaining whole
videos in memory. `--top`, `--workers`, and `--output` control selection, rendering
processes, and destination. Repeated leaders reuse the same video. Re-running an
interrupted export reuses completed traces and videos. No model calls are made.

## Shoe exhaustion

A fixed, conservative reserve guarantees enough cards to complete even split
rounds. It depends only on the configured deck size and split limit, never on
the hidden remaining composition. No cards are replaced or shuffled mid-round.

`env.cut_card` is the effective number of dealt cards that ends a shoe
after settlement. At default rules and 75% requested penetration, it is 21 for
one deck, 61 for two, 234 for six, and 312 for eight. Thus small shoes can stop
earlier than requested; the final round can also cross the cut card. Six- and
eight-deck defaults retain the requested 75% cut. Disabling splits reduces the
reserve. There is no terminal signal between ordinary rounds.

## Held-out replay

The saved-run exporter can visualize unseen seed panels independently of search:

```sh
./scripts/run examples.blackjack_videos runs/blackjack-long \
  --split holdout --top 1 --output runs/blackjack-heldout-videos
```

This uses `heldout_seeds` from `experiment.json`; `--seeds 1000 1001` overrides
them. Seeds must be disjoint from the saved training seeds. `--split validation`
is an alias for held-out seeds in Blackjack. Leaders remain selected by their
training rankings, and held-out results are written only to the export directory.
Use `--generation N` to inspect one frozen generation. The index shows training
and replay means separately, without expecting held-out scores to match training.

## Performance and verification

```sh
./scripts/run examples.blackjack --hands 100000 --repeats 3 --compare-gym
.venv/bin/python -m unittest tests.test_blackjack
```

The benchmark policy always bets and reports completed initial-wager rounds, **not steps**, and does
not inflate throughput by counting split hands separately. Timing includes
policy decisions, observation creation, action validation, settlement, and all
subsequent shoe shuffles/resets. It excludes imports and first construction/reset.
It exits unsuccessfully if the finite-shoe median falls below 10,000 hands/second;
override with `--min-hands-per-second`. `--decks` selects shoe size.

This is in-process environment throughput. Docker transport, policy generation,
neural inference, and the asynchronous runner's validation/copying add costs and
are not included. The Gymnasium comparison has different rules and fewer actions
per round; it is a reference workload, not an identical simulation.

The initial [recorded benchmark](blackjack-benchmark-2026-09-19.json), before adding
the sit-out action, on 2026-09-19,
macOS arm64 / Python 3.14.2 / Gymnasium 1.3.0, measured **122,726 hands/second**
median across three 100,000-round runs with six decks. The built-in Gymnasium
reference measured 31,042 hands/second. These are local measurements, not a
hardware-independent guarantee.

Tests cover deterministic deals, natural payouts, ace handling, doubling,
splitting, soft-17 rules, hidden-card isolation, legal actions, finite shoes,
Gymnasium's checker, space compatibility, and the existing episode runner.

## Existing environments reviewed

- [Gymnasium Blackjack-v1](https://gymnasium.farama.org/environments/toy_text/blackjack/)
  uses replacement draws from an infinite deck, hit/stand actions, and a three-value
  observation. Its current [implementation](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/blackjack.py)
  ends each episode after one hand. Changing only the draw function would still
  leave betting, card exposures, and shoe-long policy memory unimplemented.
- [PufferLib's current Ocean tree](https://github.com/PufferAI/PufferLib/tree/5.0/ocean)
  contained no blackjack environment when checked on 2026-09-19. Its current
  [documentation](https://puffer.ai/docs.html) describes native C/CUDA environments
  and a custom binding interface. That is a possible backend for much larger
  throughput targets, but unnecessary for the measured 10,000-hand requirement
  and not a drop-in replacement for this repository's Gymnasium policy runner.
