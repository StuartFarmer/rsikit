# Research algorithms

The packaged `research` namespace contains experimental optimizer implementations,
CLI adapters, and optional integrations. The shared `rsikit` library owns policies,
episodes, evaluation, process execution, and runs. Start with the
[Guide](../docs/src/guide/introduction.md), [API reference](../docs/src/api/index.md), or
[two featured examples](../docs/src/examples/index.md).

| Python package | Unified CLI selector | Checkout example |
| --- | --- | --- |
| `research.alphaevolve.paper`, `.original`, `.improved` | `alphaevolve --variant paper\|original\|improved` | `examples.alphaevolve` |
| `research.shinkaevolve` | `shinka` | `examples.shinkaevolve` |
| `research.elitesearch` | `elite` | `examples.elitesearch` |
| `research.lineagesearch` | `lineage` | `examples.lineagesearch` |
| `research.eoh` | — | `examples.eoh` |
| `research.reevo` | — | `examples.reevo` |
| `research.promptbreeder` | — | `examples.promptbreeder` |
| `research.evox` | — | `examples.evox` |
| `research.gepa` | — | `examples.gepa` |
| `research.stop_optimizer` | — | `examples.stop_optimizer` |

The [optimizer reference](../docs/src/api/optimizers.md) gives public classes and
configuration objects; the [CLI reference](../docs/src/api/cli.md) covers flags and
YAML. Run checkout examples with `python -m examples.NAME --help`. They have
algorithm-specific prerequisites and budgets; their inclusion does not establish
paper-equivalent results or mean every optional workflow is release-validated.

## Shared contract

Optimizers expose `done`, `best`, `await propose()`, and
`update({policy_id: {seed: Episode}})`. Proposals are `rsikit.PolicyDefinition`
objects. `await rsikit.search(agent, evaluate)` drives complete rounds using an
async callback, commonly bound to `Run.evaluate`. Policy failures remain episode
feedback; infrastructure errors propagate. Specialized algorithms can require
additional training evaluators or traces and return richer objects from their
own `run()` methods.

For direct Python integration, import the algorithm from its package, for example
`from research.elitesearch import Config, EliteSearch`. Configure Slick's template
root as the corresponding example or CLI adapter does. Algorithms own prompts,
repair strategies, selection, budgets, records, and checkpoints. See
[runs and recovery](../docs/src/guide/runs.md) for supported resume paths; saving a run
is not universal optimizer checkpointing.

## Execution and dependencies

Native evaluation uses POSIX process workers; it can run directly on macOS/Linux
or inside the [application container](../docs/src/guide/installation.md#docker-launcher).
`Executor` does not start Docker containers per policy. Specialized integrations
such as STOP's example have their own runtime requirements, and Ocean requires
native installation. Generated Python is executable code, not hostile-code
sandboxed input. Offline featured examples need no credentials; paid model
searches require the provider extra and API key.

Core code must not import `research` or `examples`. Research algorithms must not
import another algorithm; variants within one algorithm may share internals.
Keep general evaluation/orchestration in `rsikit` and experiment-specific scoring
with the experiment. Check this boundary with:

```sh
python -m unittest tests.test_package_boundaries -v
```

Historical plans and experiment reports live in the
[maintainer archive](../maintainer/archive/2026-10-08/docs/). They are dated research
records, not current usage instructions. Algorithm-specific attribution remains
beside source, including [EvoX provenance](evox/SOURCES.md); see the project
[NOTICE](../NOTICE) for third-party terms.
