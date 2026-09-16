# Citation verification for RSIKit conceptual modules report

Verification date: 2026-09-16. Reviewed `outputs/rsikit-improvement-modules-comparison.md` against all three research notes (`rsikit-archive.md`, `rsikit-population.md`, `rsikit-feedback.md`) and local source. This is a citation/mechanism check, not a new empirical evaluation.

## Checks performed

- Parsed every Markdown destination in the report. All existing agent, RSIKit, helper, README, NOTICE, and research-note paths resolve relative to `outputs/`. The provenance destination was initially missing while the lead was creating it; the final check found all 59 unique local destinations present (zero missing). There were 67 unique external URLs.
- Counted 45 named agent rows in section 3, consistent with the declared 12 archive-group + 16 population-group + 15 feedback-group + PersonaEvolve/LATS scope. Each row links its local implementation; 44 named-method rows also have a primary external source, while generic Optimizer is correctly identified as a local baseline without separate research attribution.
- Compared every named-method row against its research-note mechanism and fidelity caveats. No unsupported empirical rankings or benchmark reproduction claims found. Recommendations and composition boundaries are explicitly identified as synthesis.
- Reused prior direct source reads for all 12 archive-group methods, including full local loops, helpers and prompts. Independently spot-checked `rsikit/strategies.py`, `rsikit/population.py`, `rsikit/proposer.py`, `rsikit/repair.py`, `evoprompting/agent.py:207`, `meoh/agent.py:166`, `expel/agent.py:264`, `lats/agent.py:119`, and `personaevolve/agent.py:37` against cited claims.
- Confirmed RSIKit distinguishes history/pending/best and stores selection ancestry separately; serial AlphaEvolve and DGMArchive limitations are correctly stated. Repair preserves original-parent boundaries on each revision; infrastructure errors propagate. ExpeL uses exact task-embedding inner product. MEOH offspring immediately enter the parent pool. LATS no-simulation mode backs up measured reward. PersonaEvolve only rewrites configured fields.
- Confirmed report external URLs match source identities in the research notes. Prior team research is reused for population and feedback primary-page verification; this pass does not claim every historical upstream file was freshly downloaded.

## Citation edits applied

1. Corrected EvoX scorer URL commit from invalid mixed hash `0d932b690670a7e544ef5104a7cc0f3bdc4ab9b4fd2` to locally documented `0d932b690670a7e544388ad362e9876c8bd256a0`.
2. Linked the GPL-2.0-or-later statement directly to `../optimizing_the_optimizer/cmsa.py`, whose lines 3–5 carry the source-derived license notice. The existing agent source link alone did not support that helper-specific statement as directly.

No implementation code or report prose was changed by this verifier. SOURCE_LIST was left to the lead. The lead corrected the one identified wording issue: EvoPrompting trains on accepted current-generation children excluding selected next parents; it does not train on those selected parents.

## Primary-source access and evidence limits

| Source | Verification status and implication |
| --- | --- |
| [AlphaEvolve paper](https://arxiv.org/html/2506.13131v1) and [official results repository](https://github.com/google-deepmind/alphaevolve_results) | Retrieved earlier this investigation. Paper supports ensembles, edit formats, rich context, staged evaluation and separate prompt evolution; repository explicitly states it does not contain runnable agent code. Local exact archive/control choices remain adaptations. |
| [AdaEvolve paper](https://arxiv.org/html/2602.20133v1), [EvoX paper](https://arxiv.org/html/2602.23413v1) | Retrieved and method equations/algorithms inspected. Both corrected pinned SkyDiscover GitHub files and equivalent raw URLs returned cache misses in this pass. This is a retrieval limitation, not proof the commits or files do not exist; use paper plus inspected local provenance for detailed port claims. |
| [DGM pinned outer loop](https://github.com/jennyzzt/dgm/blob/a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2/DGM_outer.py) | Freshly retrieved, recognizable source file at expected commit. Local implementation and source mapping establish recursive executor boundary; report distinguishes it from DGMArchive. |
| [STOP official driver](https://raw.githubusercontent.com/microsoft/stop/main/run_improver.py) | Retrieved earlier and inspected directly. Nonzero checked utility and fallback to previous executable are explicit; no elitist incumbent comparison should be attributed to the outer loop. |
| [QUBE primary manuscript](https://openreview.net/pdf?id=y8Ctbiriia) and [indexed manuscript](https://openreview.net/pdf/0a690b7b53274608454fe2f9b6945555d56544c6.pdf) | Search-indexed primary text corroborates quality/uncertainty and offspring-aware resets. Full opens return browser verification. Medium exact-formula confidence is already disclosed. Do not claim full independent formula verification. |
| [Eureka paper](https://arxiv.org/html/2310.12931v1) and [pinned file](https://github.com/eureka-research/Eureka/blob/9eee42808b52d8abc14845d1547bfde886403ccc/eureka/eureka.py) | Paper reward-reflection and prior-batch-context passages retrieved. Raw pinned file returned cache miss; GitHub page retrieved but does not provide a complete source body through this browser extraction. Local source supports the detailed failure handling. |
| [PINSKY paper](https://arxiv.org/pdf/2007.08497), [official repository](https://github.com/aadharna/UntouchableThunder) | Paper and repository page retrieved. Earlier pinned raw implementation fetch failed. Local source + documented pinned mapping support exact generic adaptation details; no game-result reproduction established. |
| [Zero-shot selection paper](https://arxiv.org/html/2607.23505v1), [supplement repository](https://github.com/hengzhe-zhang/ppsn2026-zero-shot-llm-selection) | Paper retrieved and zero-shot no-feedback contract explicitly inspected. Supplement repository fetch returned cache miss in this pass. Numerical selector implementations remain pseudocode interpretations, as the report says. |
| [GI 2023 artifact](https://doi.org/10.5281/zenodo.8304433), [GI journal artifact](https://doi.org/10.5281/zenodo.13381774), pinned Gin links | DOI opens rejected by browser tool as unsafe to open; pinned GitHub source/tree returned cache misses. These URLs match local README provenance but this pass does not establish retrieved artifact equivalence. Report already qualifies original-source access. |
| [Optimizing the Optimizer repository](https://github.com/camilochs/optimizing-the-optimizer), [paper](https://annals-csis.org/Volume_43/drp/pdf/1481.pdf) | Repository and primary paper record retrieved earlier. CMSA formulas/license details derive from local code and explicit source-provenance record; helper code license is now directly cited. |
| Remaining population/feedback papers | Method identities and central claims were checked by their assigned researchers and are documented in their notes. This pass compared report claims to those notes and relevant local code; it does not upgrade reported historical-source gaps to fresh upstream verification. |

## Remaining required checks for lead

- The linked provenance file now exists and all local links resolve. Finish SOURCE_LIST and rerun the relative-link check if final edits introduce new destinations.
- Carry the SkyDiscover/supplement retrieval limitations above into provenance alongside the already disclosed QUBE, Gin, TAUCHI and EoT limitations.
- Test-count and Git-revision statements depend on the lead's run records; this citation pass did not rerun tests or independently certify those operational results.
- No remaining mechanistic citation mismatch identified after the EvoPrompting clarification and EvoX URL correction. No additional prose changes requested.
