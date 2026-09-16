# RSIKit improvement modules — investigation plan

## Questions
- What does each local improvement method change, and what state survives an iteration?
- Which mechanisms are shared, which innovations are distinct, and which combinations conflict?
- Which conceptual blocks already exist in RSIKit, and what is the smallest useful extraction order?

## Scope and evidence
Start with evolutionary/code improvement and adjacent reflection/memory methods; offer the full prompt-optimization collection as an explicit scope option. Inspect implementations and prompts, cross-check original primary sources, and distinguish adaptations from original algorithms. Reasoning-only methods are contextual, not automatically self-improvement.

## Deliverable
One cited comparison: outputs/rsikit-improvement-modules-comparison.md, with per-agent matrix, conceptual module contracts, composition recipes, current RSIKit mapping, gaps, and prioritized experiments. Supporting evidence and provenance stay in outputs/.research and the provenance sidecar. No implementation changes or paid evaluations.

## Task ledger
| ID | Owner | Task | Status |
|---|---|---|---|
| T1 | researcher A | Archive, adaptive, recursive code evolution | complete: .research/rsikit-archive.md |
| T2 | researcher B | Population and multiobjective code evolution | complete: .research/rsikit-population.md |
| T3 | researcher C | Prompt evolution, reflection, experience and learning | complete: .research/rsikit-feedback.md |
| T4 | lead | RSIKit baseline, scope inventory, synthesis | complete: 45 methods, 12 conceptual modules |
| T5 | verifier / reviewer | Evidence and module-boundary review | complete: citation verification and conceptual review; corrections resolved, access caveats retained |

## Acceptance
Every in-scope package has an explicit mapping; proposed modules specify input/output/state and distinctions; primary-source claims have citations; no unmeasured quality ranking; no universal framework proposed without a concrete consumer.

## Decisions
The user authorized the investigation; proceed with reversible research and documentation without an additional approval gate. User confirmed evolutionary/code agents plus relevant reflection and memory; do not survey the full prompt-optimization catalogue. Source-comparison/deep-research skill instructions authorize parallel researchers and verification. No implementation work requested or performed.

## Verification log

| Claim | Check | Result |
|---|---|---|
| Existing RSIKit baseline | `rtk proxy optimizer/.venv/bin/python -B -m unittest discover -s rsikit/tests` | 38 discovered; 32 passed, six Docker integration skips; exit 0 |
| Scope coverage | Parse section 3 agent rows and distinct local package links | 45 rows / 45 distinct packages |
| Architectural fit | Independent review of report against source, RSIKit state and examples | No blocking issue; corrected EvoPrompting exclusion wording and EvoX citation |
| Paper fidelity | Cross-read primary papers/repositories and local adaptation notes | Qualified limitations retained for unavailable sources and deliberate local adaptations |
