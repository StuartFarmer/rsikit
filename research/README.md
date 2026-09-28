# Research algorithms

Each algorithm builds on the shared `rsikit` library and owns its search logic,
records, and prompts. Algorithms must not import another research algorithm;
generic library functionality belongs in `rsikit`; experiment-specific reward
callbacks and rollout orchestration live in shared `research` modules. The core must not import research or
examples. Variants within one algorithm may reuse that algorithm's internals.

| Package | Guide | Example command |
| --- | --- | --- |
| `research.alphaevolve` | [AlphaEvolve](../docs/ALPHAEVOLVE.md) | `python -m examples.alphaevolve` |
| `research.shinkaevolve` | [ShinkaEvolve](../docs/SHINKAEVOLVE.md) | `python -m examples.shinkaevolve` |
| `research.elitesearch` | [EliteSearch](../docs/ELITESEARCH.md) | `python -m examples.elitesearch` |
| `research.lineagesearch` | [LineageSearch](../docs/LINEAGESEARCH.md) | `python -m examples.lineagesearch` |

Run these commands from the repository root after following the
[setup instructions](../README.md#setup). Searches require the Docker worker and
an API key and make paid model calls. Research code is included in the source
distribution; the installable library wheel contains only `rsikit`.

Imports now use the `research` namespace, for example:

```python
from pathlib import Path
from slick import prompts

from research import elitesearch
from research.elitesearch import Config, EliteSearch

prompts.TEMPLATE_ROOT = Path(elitesearch.__file__).parent / "prompts"
```

Shared modules available to every algorithm and runner:

- `rsikit.policy.validate_policy`: explicit source checks after policy generation.
- `rsikit.Evaluator` and `rsikit.Episode`: rollout execution and raw trajectories.
- `research.rollouts.Rollouts`: execution, episode persistence, and experiment-local reuse.
- `research.rewards`: cumulative-reward fitness callbacks and per-seed measurements.
  AlphaEvolve owns its richer `EvaluationResult` and screening in its own package.
- `rsikit.Policy.from_text` / `from_file` and `to_text` / `to_file`: canonical solution
  loading and saving, preserving source and identity without host execution.
- `rsikit.envs.tasks`: environment presets and `make_environment`.
- `rsikit.progress`: `ProgressHandler` and `show_scores`.
- `rsikit`: `Policy`, `Run`, and `Executor` with Docker evaluation.

Each optimizer composes its own `SelfHealer` in `healing.py`, with task context,
provider, and a local repair prompt. It proposes a repair; the optimizer owns
retry budgets, validation, and candidate acceptance. AlphaEvolve variants share
the original variant's healer. Prompt operations use Slick's `render` and `parse`
with the provider's `acall`; raw responses go into optimizer attempt records
before parsing, including malformed responses.

Generation and healing operations return `type[Policy]`. Their private response
schemas and mutation contracts live in each algorithm's `generation.py`.
Optimizers validate generated policy source explicitly before accepting proposals;
loading a policy does not validate it. AlphaEvolve, ShinkaEvolve, and LineageSearch
own their protected-region rules. EliteSearch edits the whole organism and gives
evolution-marker comments no special meaning.

The dependency boundary is checked with
`python -m unittest tests.test_package_boundaries -v`.
