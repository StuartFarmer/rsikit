# Research optimizers

These packaged research implementations build on the shared
[optimizer protocol](optimization.md). Treat them as experimental algorithms with
explicit configuration, not promises of paper-equivalent results or API stability.
Use the [Guide](../guide/optimization.md) for feedback/budget semantics and
[recovery table](../guide/runs.md#what-resume-means) before planning a resumable run.

## Entry points

| Implementation | Public Python import | Unified CLI | Search mechanism |
| --- | --- | --- | --- |
| AlphaEvolve, paper variant | `research.alphaevolve.paper.AlphaEvolve`, `Config` | `alphaevolve --variant paper` | Persistent niche archive, multiple objectives, migration. |
| AlphaEvolve, original baseline | `research.alphaevolve.original.AlphaEvolve`, `Config` | `alphaevolve --variant original` | Local island-based baseline. |
| AlphaEvolve, improved baseline | `research.alphaevolve.improved.AlphaEvolve`, `Config` | `alphaevolve --variant improved` | Revised local baseline; distinct from the paper archive. |
| EliteSearch | `research.elitesearch.EliteSearch`, `Config` | `elite` | Fixed elite population, new proposals, edits, and remixes. |
| ShinkaEvolve | `research.shinkaevolve.ShinkaEvolve`, `Config` | `shinka` | Island search with archives, novelty, and migration. |
| LineageSearch | `research.lineagesearch.LineageSearch`, `Config` | `lineage` | Experimental families, frontier search, and stagnation stopping. |
| EoH | `research.eoh.EoH` | Python / `examples.eoh` | Five operators evolving descriptions and implementations. |
| ReEvo | `research.reevo.ReEvo`, `Config` | Python / `examples.reevo` | Pairwise and accumulated reflection. |
| PromptBreeder | `research.promptbreeder.PromptBreeder` | Python / `examples.promptbreeder` | Policy and mutation-instruction co-evolution. |
| EvoX | `research.evox.EvoX`, `Config` | Python / `examples.evox` | Policies and executable search selectors. |
| GEPA adaptation | `research.gepa.GEPA` | Python / `examples.gepa` | Reflection using training evidence and separate selection coverage. |
| STOP adaptation | `research.stop_optimizer.STOP`, `Problem` | Python / `examples.stop_optimizer` | Evolving executable improvers with bounded capabilities. |

`examples.*` modules require a checkout. Only the first four selector names are
accepted by the unified CLI; a Python/example entry point does not imply CLI
support. Specialized algorithms may require extra callbacks such as traces or
training evaluators; inspect their constructors below. Their generated selectors
or improvers execute Python, under the same execution-boundary limitations as
[policies](../guide/installation.md#docker-launcher).

When integrating directly in Python, configure Slick's template root as the
corresponding example or adapter does. The AlphaEvolve variants share the
`research/alphaevolve` root; other optimizers use their local `prompts` directory.
The shared `search` return value is a definition or `None`; algorithm-specific
`run()` helpers can instead return richer result objects.

## AlphaEvolve variants

The `paper` name identifies this repository's implementation of published
mechanisms with local archive rules. It is not an upstream Google package.
Its archive maximizes every metric; `objective` chooses the global winner.
`EvaluationResult` supports explicit metrics/descriptors for specialized callers;
ordinary `search` integrations still pass episode mappings to `update`.

<!-- api: research.alphaevolve.paper.agent.AlphaEvolve
{inherited_members: true, members: [done, best, propose, update, checkpoint, close,
    register_initial]}
-->

<!-- api: research.alphaevolve.paper.agent.Config
{inherited_members: true, members: true}
-->

<!-- api: research.alphaevolve.paper.evaluation.EvaluationResult
{inherited_members: true, members: true}
-->

<!-- api: research.alphaevolve.original.agent.AlphaEvolve
{inherited_members: true, members: [done, best, propose, update]}
-->

<!-- api: research.alphaevolve.original.agent.Config
{inherited_members: true, members: true}
-->

<!-- api: research.alphaevolve.improved.agent.AlphaEvolve
{inherited_members: true, members: [done, best, propose, update]}
-->

The improved variant re-exports the original variant's `Config`; it has no
separate configuration class.

## EliteSearch

<!-- api: research.elitesearch.agent.EliteSearch
{inherited_members: true, members: [done, best, propose, update, restore, records]}
-->

<!-- api: research.elitesearch.agent.Config
{inherited_members: true, members: true}
-->

## ShinkaEvolve

<!-- api: research.shinkaevolve.agent.ShinkaEvolve
{inherited_members: true, members: [done, best, propose, update]}
-->

<!-- api: research.shinkaevolve.agent.Config
{inherited_members: true, members: true}
-->

## LineageSearch

<!-- api: research.lineagesearch.agent.LineageSearch
{inherited_members: true, members: [done, best, propose, update]}
-->

<!-- api: research.lineagesearch.agent.Config
{inherited_members: true, members: true}
-->

## Additional experimental implementations

These integrations have specialized research entry points and no unified CLI
resume contract. EoH, PromptBreeder, GEPA, and STOP configure through constructor
arguments rather than a public `Config` class. Sequential optimizers expose
`aclose()` to close a suspended proposal stream; their local generator state is
not a durable checkpoint.

<!-- api: research.eoh.agent.EoH
{inherited_members: true, members: [done, best, propose, update, run, aclose]}
-->

<!-- api: research.reevo.agent.ReEvo
{inherited_members: true, members: [done, best, propose, update, run, aclose]}
-->

<!-- api: research.reevo.agent.Config
{inherited_members: true, members: true}
-->

<!-- api: research.promptbreeder.agent.PromptBreeder
{inherited_members: true, members: [done, best, propose, update, run, aclose]}
-->

<!-- api: research.evox.agent.EvoX
{inherited_members: true, members: [done, best, propose, update, run, aclose]}
-->

<!-- api: research.evox.agent.Config
{inherited_members: true, members: true}
-->

<!-- api: research.gepa.agent.GEPA
{inherited_members: true, members: [done, best, propose, update, run, aclose]}
-->

<!-- api: research.stop_optimizer.agent.STOP
{inherited_members: true, members: [done, best, propose, update, run, aclose]}
-->

<!-- api: research.stop_optimizer.agent.Problem
{inherited_members: true, members: true}
-->

