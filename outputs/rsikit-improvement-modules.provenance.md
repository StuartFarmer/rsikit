# Provenance: RSIKit improvement modules

- Date: 2026-09-16.
- Final comparison: [rsikit-improvement-modules-comparison.md](rsikit-improvement-modules-comparison.md).
- Scope confirmed by user: evolutionary/code agents plus relevant reflection and memory methods. Selected prompt, training and diagnostic methods are included only as mechanism donors or boundary examples; this is not the full prompt-optimization catalogue or a global survey of every published system.
- Coverage: 45 unique local implementations, plus the four current RSIKit strategies. Twelve conceptual module responsibilities, per-method mappings, composition recipes and an ordered extraction recommendation.
- Research rounds: one parallel evidence-gathering round, followed by citation verification and independent synthesis review.
- Source snapshot: checkout HEAD `46bfbaa42d17c7564a567d149bd63a1963441c7f`. RSIKit and outputs were already untracked, so HEAD alone does not pin their source state. Read current local files, including source, README qualifications, relevant helpers and prompt templates. No implementation changes were made.
- Evidence hierarchy: current local implementation for current mechanics; primary papers/author repositories for method intent and attribution; local README historical-source mappings where fresh upstream retrieval was unavailable. Module boundaries, combinations and build order are explicitly architectural synthesis.
- Sources: 67 unique external URLs cited in the final comparison, alongside local files. Not every URL is a successfully re-fetched full-text source; unavailable references remain explicitly qualified provenance pointers. Supporting notes contain additional related source links.
- Sources rejected as support: no secondary benchmark rankings or unverified performance transfers were used. Unavailable upstream internals were not used to claim exact implementation fidelity.
- Plan: [research plan](.plans/rsikit-improvement-modules.md).
- Evidence: [archive/adaptive/recursive research](.research/rsikit-archive.md), [population research](.research/rsikit-population.md), [feedback/memory research](.research/rsikit-feedback.md).
- Review artifacts: [citation verification](.research/rsikit-citation-verification.md), [independent review](.research/rsikit-review.md).

## Verification performed

Fresh local command:

```sh
rtk proxy optimizer/.venv/bin/python -B -m unittest discover -s rsikit/tests
```

Exit status 0: 38 discovered, 32 passed, six Docker integration tests skipped. No live model, fine-tuning, Docker integration or benchmark-quality experiment was run. Existing tests substantiate the current RSIKit lifecycle and scripted behavior, not the performance of proposed combinations.

The lead checked the final report's local link targets and counted 45 distinct per-agent mappings. Citation and conceptual reviewers separately checked sources, wording and scope. **Final outcome: PASS WITH NOTES.** No blocking architecture or citation mismatch remains; upstream access limitations remain explicitly qualified. Corrected the EvoX pinned-source URL and clarified that EvoPrompting excludes selected next parents from current-round training data.

## Material source limitations

- AlphaEvolve's public results repository is not the original agent implementation. Exact local sampling, archive and meta-credit behavior is described as adaptation; FunSearch-derived initialization/reseeding is attributed separately.
- QUBE's full primary manuscript encountered browser verification. Indexed primary snippets corroborated the broad mechanism; exact primary formula fidelity remains medium confidence, while the local code was inspected directly.
- Gin / LLM-GI conference and journal source URLs could not be freshly fetched in the population pass. Local code and documented artifact mapping support the comparison, not a new upstream audit.
- Some pinned PINSKY/Eureka raw-file fetches failed; local mechanisms and available primary papers were used, with adaptation qualifications.
- TAUCHI-GPT lacks a verified original source tree in the local provenance; the port combines described system features. The comparison does not claim persistent cross-run learning.
- Promptbreeder's minimal co-author implementation is later than the original paper experiments. EoT's local provenance records inaccessible original code. Zero-shot selection utilities are interpreted rather than verified original generated selectors.
- EoH, EoH-S, MEOH and other ports deliberately resolve paper/released-code differences. The comparison describes the actual local choices and does not imply uniform paper fidelity.
- No relative performance ranking, guaranteed improvement, faithful rationale claim or optimizer-generalization claim is inferred from the source code.
- The final citation pass also encountered cache-miss failures on corrected pinned SkyDiscover files/raw equivalents and the zero-shot selection supplement repository. Gin DOI browser opens were rejected and pinned-source fetches failed. Those references identify documented provenance; detailed local behavior is supported by inspected local source. DGM's pinned outer loop and the PINSKY repository were retrieved successfully in that pass.
