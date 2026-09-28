# EliteSearch

A separate optimizer with a fixed leaderboard and a fresh population each generation.
It does not use LineageSearch's families, decomposition, culling or stagnation rules.

- **10 elite slots**, filled from the best unique, successfully evaluated organisms.
- **50 candidate attempts per generation**, in addition to the retained elites.
- Generation 1 invents 50 new ideas. Subsequent generations default to 10 new
  ideas, 20 focused edits of one elite, and 20 remixes of up to three distinct elites.
- Each generation uses a frozen snapshot of the prior leaderboard. Remixes receive
  all selected parents' source code, descriptions and measured scores; edits use
  exact search/replacement operations. With only one elite, remix slots become edits;
  with no elites, every slot invents a new idea.
- After evaluation and repair, rank the old elites and scored newcomers together
  by mean reward and keep the top 10. Equal scores favor current-generation edits, then remixes, then existing elites,
  then new proposals; ties within a category favor the lower organism ID. A
  candidate tying the best can still displace a lower-ranked elite. There are no
  resets or percentage culls, and low-scoring newcomers cannot evict better elites.

The leaderboard fills as valid unique candidates become available. It can have fewer
than 10 members if generation or evaluation fails. Invalid generations and runtime
errors share `--max-repairs` per organism; exhausted attempts are discarded rather
than silently adding more population slots. AST-equivalent programs are repaired as
duplicates, so renames and comments cannot occupy duplicate leaderboard slots.

All candidates use the same search seeds. Nonfinite scores, missing results or
changed seed panels abort the run. Infrastructure/provider failures propagate;
they do not become low fitness. Failed source, raw model output, repair revisions,
parent IDs and measured results remain in the run database.

## Run

Use the existing OpenRouter setup and scientific Docker worker from the README.
This command makes paid model calls; no additional Docker rebuild is required:

```sh
.venv/bin/python -B -m examples.elitesearch \
  --env BipedalWalker-v3 \
  --elites 10 --population 50 --generations 20 \
  --new-fraction 0.2 --remix-fraction 0.4 --remix-parents 3 \
  --generation-concurrency 100 --concurrency 8 --max-repairs 5
```

`--new-fraction` and `--remix-fraction` each allocate a fraction of population slots,
rounded down. The remaining slots edit elites. Their sum must be at most 1.
`--generations` bounds the run (default 20, or 1,000 population attempts). Elite
parents are sampled uniformly without replacement for a remix. `--search-seed`
controls operator shuffling and parent sampling.

Generation overlaps evaluation as candidates become ready. All model calls,
including repairs, share `--generation-concurrency`; Docker episode workers use
the separate `--concurrency` setting. New candidates can use free workers while
earlier candidates finish their remaining seeds. A generation completes before breeding the
next one so it can use the updated leaderboard.

`--generation-timeout` sets both the OpenRouter request timeout and the search's
wall-clock deadline per model call (default 120 seconds), including repairs.
For a model that needs longer, try `--generation-timeout 600`; lowering
`--generation-concurrency` reduces simultaneous requests. Waiting for a model-call
slot does not consume its deadline. `--episode-timeout` separately limits execution
of a generated policy in Docker. A generation timeout still aborts and records the
run; it is not treated as a bad policy or automatically retried.

Rich shows generations, evaluated/discarded population slots, filled elite slots,
generation/evaluation progress and the leaderboard after each generation.
Artifacts include `experiment.json`, `run.log`, `leaderboard.json`, `best.py`,
`summary.json`, and database tables `elitesearch_organism` and
`elitesearch_generation`. Every generation stores its elite IDs and promotions.
These are inspection records; resuming interrupted searches is not implemented.

Search seeds default to 0–4. The final best elite is also evaluated on held-out
seeds 100–104, without changing the leaderboard or repairing against those results.
Improvement on search seeds is not a guarantee of improvement on unseen seeds.

Programmatic callers import `Config` and `EliteSearch` from
`research.elitesearch`, configure Slick's template root to `research/elitesearch/prompts`, and inject
an async evaluator returning `{policy.id: EvaluationResult(seed_scores=per_seed_scores)}`.
Import `EvaluationResult` and the reusable `evaluate_gym(run, policies, seeds=...)`
adapter from `rsikit`. Candidate failures use `EvaluationResult(failure="diagnostic")`;
screening rejections use `accepted=False` without a failure and are discarded without
repair. Infrastructure failures raise. The old `Measurement(scores, failure=None)`
constructor remains compatible.
Evaluation callbacks run concurrently; a shared Run/Executor bounds episode workers.
The caller owns isolated execution and persistence through `on_checkpoint` and
`agent.records()`. Generated code is never executed by the optimizer itself.

See [evaluation throughput](EVALUATOR_PERFORMANCE.md) for timing and benchmark commands.
