# ShinkaEvolve

`shinkaevolve/` is a separate optimizer alongside `alphaevolve/`. It ports the
previous local implementation on `main` to the current Policy/Run API. It uses
Slick, API models, and Gymnasium; it does not train model weights or depend on
the upstream ShinkaEvolve framework. See [provenance](../shinkaevolve/NOTICE).

## Run an experiment

Use the existing OpenRouter/Box2D installation and Docker worker described in the
[README](../README.md). No image rebuild is needed for this optimizer alone.
With `OPENROUTER_API_KEY` exported:

```sh
.venv/bin/python -B -m examples.shinkaevolve --env LunarLander-v3 --generations 10 --batch-size 25 --seeds 0 1 2 3 4 5 6 7 8 9
.venv/bin/python -B -m examples.shinkaevolve --env BipedalWalker-v3 --generations 10 --batch-size 25 --seeds 0 1 2 3 4 5 6 7 8 9
```

The environment presets are shared with AlphaEvolve: continuous LunarLander with
wind, normal BipedalWalker, and CartPole. Native episode limits apply unless
`--max-steps` is supplied. Every policy gets the same specified evaluation seeds;
selection maximizes their mean reward. Keep additional seeds for held-out replay.

The default model is `openai/gpt-oss-120b:nitro`. Repeat `--model MODEL` to supply
an ensemble; adaptive allocation has no effect with only one model. Models are
indexed in the supplied order in SQLite and named in `experiment.json`.
Generation and evaluation each default to concurrency 4, controlled separately
by `--generation-concurrency` and `--concurrency`.

Rich progress shows generated names, descriptions, repairs, and scores. Each run
gets its own SQLite database, Python exports, `run.log`, and `experiment.json`.
Videos use the existing replay command and do not need model calls:

```sh
.venv/bin/python -B -m examples.replay runs/YOUR_RUN --env LunarLander-v3 --top 3 --seeds 10 11 12
```

## Python loop

```python
from pathlib import Path
from slick import prompts
from slick.providers import OpenRouterAPI
import shinkaevolve
from shinkaevolve import Config, ShinkaEvolve
from examples.alphaevolve import make_environment
from rsikit import Executor, Run

# Configure Slick once at application startup.
prompts.TEMPLATE_ROOT = Path(shinkaevolve.__file__).parent / "prompts"
provider = OpenRouterAPI(model="openai/gpt-oss-120b:nitro")

# Inside an async function:
with make_environment("LunarLander-v3") as env:
    generator = ShinkaEvolve(
        task="Maximize cumulative episode reward.",
        context=env.instructions,
        provider=provider,
        config=Config(islands=2),
    )
    with Run.create(name="shinka-lander", environment=env, executor=Executor()) as run:
        for _ in range(10):
            policies = await generator.generate(n=25, concurrency=4)
            scores = await run.evaluate(policies, seeds=[0, 1, 2])
            generator.update(scores)
            run.save(*generator.records(seeds=[0, 1, 2], complete=True))
```

This shows the success path. `examples.shinkaevolve.run_search` adds runtime
repair, saving incomplete batches, logging, and progress. It calls
`generator.evaluation_failed(failures)` and `await generator.repair(policy,
diagnostic)` for sandbox policy errors. Repair returns a replacement Policy or
`None` when that candidate is discarded. Provider and infrastructure errors
remain run errors rather than being treated as low fitness.

## Search behavior

| Mechanism | Default |
| --- | --- |
| Islands | 2 bounded populations, capacity 40 each |
| Archive retention | Best 30% of capacity, remaining slots sampled randomly |
| Parent selection | Fitness sigmoid relative to island median, divided by `1 + offspring`; also `uniform`, `best`, and inverse-rank `power` |
| Inspirations | Up to 2 top nonparents plus 4 random others from that island |
| Proposal operations | 45% diff, 45% full rewrite, 10% crossover; crossover falls back to rewrite without a second parent |
| Model allocation | Sample untried models first, then sample from normalized positive improvement plus UCB exploration weights |
| Migration | Every 10 terminal attempts, copy 10% of each population around a ring, excluding its best member |
| Reflection | At batch boundaries after crossing each 10-attempt interval; retain up to 5 recommendations |
| Resampling | Up to 3 proposals per candidate attempt |
| Repair | Up to 2 repairs total per attempt across generation and execution |

Model improvement is measured over the larger of the parent's score and the
initial policy's score. It is clipped at zero, normalized by the largest gain
seen across models, transformed with `expm1`, and combined with the exploration
term. Initial proposals and discarded candidates contribute zero gain. This is
weighted sampling, not choosing the largest UCB deterministically.

Exact implementation duplicates within an island or its concurrent proposal
batch are rejected. Optional `embed=async_callable` compares mutable code regions
using cosine similarity. Above `Config.novelty_threshold` (0.95), candidates are
rejected unless an optional `novelty_provider` judges them algorithmically novel.
The CLI does not configure embeddings or a novelty judge. Reflection uses the
main provider by default; `meta_provider` overrides it, and `meta_interval=0`
disables it. Malformed reflection keeps the last valid recommendations.

All initial proposals are generated before evaluation. The first evaluated
survivor seeds every island, as the previous implementation did with its supplied
seed; other initial survivors enter their assigned islands. Subsequent batches
select from previously evaluated archives. This differs from improved
AlphaEvolve's independent island founders. Shinka's prompts use scalar mean
rewards; the Run still stores each seed's result.

Each batch attempts `n` candidates, not necessarily `n` API calls or survivors.
Resampling, repair, reflection, and optional novelty judgment add calls. Failed
repairs discard the candidate while its siblings and later generations continue.
Use call budgets as well as evaluation budgets when comparing optimizers.

## Stored history

`generator.records(...)` returns its actual SQLModel Evaluation records and the
current Generation record. `run.save(*records)` infers and creates their tables.
The CLI saves these before evaluation, after failures, and at batch completion.

- `shinkaevolve_evaluation`: generation, attempt, repair revision, island, model
  index, patch operation, parent IDs (both for crossover), policy ID, status,
  score, proposal/repair counts, error, optional novelty evidence, and model gain.
- `shinkaevolve_generation`: completion status, seeds, full island populations
  with scores, migration/reflection events, recommendations, model weights, and
  model observation counts.

Repair revisions retain earlier failed policies. Group by attempt and take the
latest revision when counting terminal outcomes; proposal/repair counts are
cumulative within an attempt. Model gain is decimal text to retain large finite
score differences. The core policy table contains code and per-seed rewards.

For a curve of the best retained score in each island:

```sql
SELECT g.number AS generation, CAST(island.key AS INTEGER) AS island,
       MAX(json_extract(member.value, '$.score')) AS best_score
FROM shinkaevolve_generation AS g,
     json_each(g.islands) AS island,
     json_each(island.value) AS member
WHERE g.complete = 1
GROUP BY g.number, island.key
ORDER BY g.number, island;
```

These records support analysis, not optimizer restoration. `Run.open` can inspect
and re-evaluate saved policies, but the CLI starts a new optimizer and run. Raw
model responses remain in memory in `generator.calls`; full response transcripts
are not stored in SQLite. This is a batched adaptation of the local implementation,
not a claim to reproduce upstream benchmark results or identical sequential traces.
