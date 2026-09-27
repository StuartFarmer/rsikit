Play no-limit Texas Hold'em, up to six players, blinds 1/2, no rake, no antes.
Every hand starts with the same configured stack. Maximize net chips per hand.
Each generation uses a fixed population, balanced table assignments, duplicate
deals with seat rotations, and equal hand counts per competitor. Scores are
big blinds won per 100 hands. Incumbents are evaluated again each generation.

Implement Solution(Policy) with async act(self, observation). Return an integer:
-1 = raise all-in (only when raising is legal, including a one-chip short bet);
0 = check when free, otherwise fold; 1 = check/call; >=2 = raise TO this many
chips total on the current betting street. Raises must be within min_raise_to
and max_raise_to, inclusive. Both are zero when raising is unavailable. An
all-in below a full minimum raise is represented by the engine's legal bounds.
Invalid actions or exceeded timeouts fail the candidate; no silent clipping.

The observation is a dict:
- seat: your physical seat index; button: dealer seat.
- street: 0 preflop, 1 flop, 2 turn, 3 river.
- hole_cards: your two strings, e.g. ["As", "Td"].
- board: zero to five public card strings.
- stacks, bets, active: per-seat remaining chips, street contributions, and
  whether the player has not folded (all-in players remain active).
- pot: chips in the pot, INCLUDING current bets.
- to_call: chips needed to call, capped at your remaining stack.
- min_raise_to, max_raise_to: legal raise-to bounds; zero if unavailable.
- big_blind: 2.
- history: [street, seat, action] for every prior decision in this hand.

The action_space is Discrete(starting_stack + 2, start=-1), but only the dynamically legal
actions described above may be returned. observation_space is an empty Dict
placeholder; use the observation fields above. No opponent cards, future cards,
deal seeds or evaluator objects are exposed. Policy randomness is independently
seeded; never infer deals from your RNG seed. Use only the Python standard
library, NumPy and Numba. No files, network, processes, environment inspection or LLM
calls. Code executes under a separate UID with a private temporary directory.

Your policy instance is recreated every hand, and its process is restarted for
every duplicate rotation. Use self.rng after calling super().reset(seed=seed).

Speed matters: target the configured mean policy-call budget (default 5 ms,
including repeated reset and act calls, with one second of cumulative grace).
Slow policies fail with timing feedback and must be optimized before evaluation.
For computationally intensive strategies, use module-level @numba.njit(cache=False)
numeric kernels on NumPy arrays. Convert card strings to integers outside JIT;
keep the observation dict and asynchronous methods outside compiled functions.
Use serial kernels: parallel=True, multiprocessing and threads are unavailable.
Avoid large Python Monte Carlo loops or repeated Python combination enumeration.
Warm up required signatures in reset; the first reset and first act in each
rotation have a separate compilation allowance (default 15 seconds each) and
are excluded from the mean speed budget. Module code and compiled kernels persist
across hands within a rotation, while each hand receives a fresh policy instance.
Do not accumulate opponent/game history in module or class globals. Disk caching
is unavailable for generated code; use cache=False.
