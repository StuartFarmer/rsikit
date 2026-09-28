# Research algorithms

Each algorithm builds on the shared `rsikit` library and owns its search logic,
records, and prompts. Algorithms must not import another research algorithm;
shared functionality belongs in `rsikit`. The core must not import research or
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

- `rsikit.generation.edits`: candidate contracts, source validation, and exact edits.
- `rsikit.generation.RecordingProvider`: raw response capture before structured parsing.
- `rsikit.EvaluationResult` and `rsikit.evaluate_gym`: shared evaluation evidence,
  candidate failures, optional screening, and cached Gym measurements.
- `rsikit.Policy.from_text` / `from_file` and `to_text` / `to_file`: canonical solution
  loading and saving, preserving source and identity without host execution.
- `rsikit.envs.tasks`: environment presets and `make_environment`.
- `rsikit.progress`: `ProgressHandler` and `show_scores`.
- `rsikit`: `Policy`, `Run`, and `Executor` with Docker evaluation.

The dependency boundary is checked with
`python -m unittest tests.test_package_boundaries -v`.
