# Paper-based AlphaEvolve

Implement the mechanisms described in sections 2.1–2.6 of
https://arxiv.org/html/2506.13131v1 . This is an independent implementation,
not a reproduction of unpublished DeepMind source or undisclosed hyperparameters.

## Requirements

- Make `alphaevolve.paper` the example's default; retain `original` and `improved`
  as explicitly local baselines.
- Replace champion-only selection with a persistent evolutionary program database:
  evaluated source, multiple maximized metrics, numeric descriptors, textual outputs,
  lineage, island membership, and MAP-Elites cells. Sample actual retained populations,
  including lower primary-score niche winners. Preserve a global primary-score best.
- Use explicit descriptor bounds/bin counts, one elite per metric per niche, rank-based
  exploitation and uniform exploration. Migrate selected members between islands
  without replacing whole populations. These details are local choices; section 2.5
  specifies the combination but not its exact rules.
- Deduplicate syntax-equivalent programs independently of display names/comments.
- Keep exact edits, immutable evolution blocks, full rewrites, model ensembles,
  prompt variants, and generated search guidance. Render all metrics, descriptors,
  evaluation output, and per-seed feedback in mutation/rewrite prompts.
- Accept evaluated initial policies; permit generated founders for the Gym example.
- Support trusted asynchronous evaluator stages with explicit thresholds and optional
  feedback graders. Only fully evaluated, accepted results enter the breeding archive.
  Threshold rejection is different from a policy crash or infrastructure failure.
- Overlap generation and evaluation through a bounded asynchronous pipeline; persist
  attempt history, evaluated programs, and population state. Do not launch paid searches
  as part of verification.
- Gym evaluation continues through RSIKit's sandbox, with mean reward, worst-seed
  reward, and negative reward standard deviation as maximized metrics. Descriptor
  bins on mean reward and seed variability are explicit performance niches, not
  claimed gait/behavior descriptors. Default bins are environment-specific local choices.
- Reject duplicate top-level `Solution` definitions through the shared AlphaEvolve
  validator, including repaired output. Existing stored runs remain readable.
- BipedalWalker defaults to full rewrites; CLI mode override supports controlled trials.

## Boundaries

The paper's domain-general evaluator is represented by callback-based evaluation in
the optimizer. The bundled runnable integration is still the Gym policy interface;
arbitrary generated programs must be executed by a caller-supplied isolated evaluator.
The installed OpenRouter model remains selectable; no claim is made to reproduce the
paper's Gemini model mix or results. LLM feedback/cascades/meta evolution are optional
as described by the paper and must have executable interfaces, not documentation stubs.

## Acceptance

Tests must demonstrate niche preservation below the global best, parent sampling from
multiple retained programs, multiobjective elites, non-destructive migration,
name-independent deduplication, persistence after reopening, complete prompt feedback,
cascade pruning, generation/evaluation overlap and cancellation, and duplicate-class
repair. Existing variant/history/repair tests remain passing. Document each mechanism
against its paper section and identify all consequential local choices.
