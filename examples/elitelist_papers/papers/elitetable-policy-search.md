
# EliteTable: Evolving Executable Policies with a Language Model

  

*Working manuscript, 25 September 2026. The main experiment is running under the frozen paper1-main-v1 protocol. Outcome tables and empirical conclusions remain pending. This editable manuscript is separate from the frozen companion copy.*

  

## Abstract

Large language models have been shown to perform well at completing tasks that would otherwise be assumed to be outside of their scope. One such domain is in automated heuristic design, also known as automated research, program generation, program optimization, etc. These methods use an LLM and evolutionary computing to generate solutions and policies to a problem which improve over trial and error. While most of these methods adhere strictly to the rough nature of evolution, throwing out many solid solutions in preference of more exploration, we designed an agent which leans on an elite policy table to sample from which creates a middle ground between pure hill-climbing approaches and overly broad search heuristics. EliteTable, our algorithm, was run on six standard RL environments and was shown to solve most in only 10 generations while providing clean and concise solutions that are completely interpretable. This is a step towards a deeper understanding of how far the standard "propose-evaluate-mutate" loop of evolutionary computing can be applied to LLMs to create automated problem solvers.


## 1. Introduction

Solutions to arbitrary problems is something that every field of study is seeking. Whether that be computer science, biology, or medicine, to have a broad 'meta algorithm' that is optimized to solve problems in general would mean the acceleration of these fields in their research and discoveries.

Recently, progress has been made in this endeavor with the help of large language models, which have been the subject of society at large ad nauseam. The subject has many names: automated algorithm design, heurstic design, program generation, policy generation, and even automated research. But the core tenet of these algorithms is the same: given any problem that can be simulated and scored, come up with a solution that scores the most.

**FINISH THIS**
Some such examples of these algorithms have been Google DeepMind's AlphaEvolve and SakanaAI's ShinkaEvolve which take their own advanced approaches to properly sampling across different past solutions, incorporating the right amount of mutation and crossover, and even prompt modification to come up with optimal solutions.
**FINISH THIS**

At two ends of the spectrum are two approaches. FunSearch maintains a single list of leading solutions that get fed back into the prompt which generates the next set of solutions. This can be seen as a form of hill-climbing which can be prone to rigidity and getting stuck in local minima. ShinkaEvolve maintains a wide set of islands which can crossover with each other, a full database of solutions that are sampled from, and richer mutations and metaoptimizations which can lead to more robust solutions at the cost of much longer search times.

EliteTable proposes something in the middle. It takes from the elitist set concept of FunSearch while also allowing for the softer evolution process of ShinkaEvolve. EliteTable maintains a set number of elite (leading) solutions which are then sampled from randomly for the next generation. The next generation proportionally proposes a third of the solutions as brand new concepts, a third as single edits and modifications of selected elites, and a third as remixes of several elites into a single solution concept.

The result is a simple optimizer that is easy to understand, generalizes decently well, and illustrates that the solution space of these 'meta algorithms' themselves is something that can be explored. There are surely myriads of alternative architectures and approaches towards these program optimizers that work as well. Understanding, and eventually ablating, these approaches will help us understand what functionality works and what doesn't when it comes to self-improvement via LLMs.

## 2. Problem statement

  

Let \(e\) denote an environment and \(p\) a policy program. During an episode, the policy receives the native observation \(o_t\) and may maintain internal state \(h_t\):

  

\[

(a_t,h_{t+1})=p(o_t,h_t).

\]

  

The evaluator, rather than the policy, accumulates the undiscounted episode return

  

\[

R_e(p,s)=\sum_{t=0}^{T(e,p,s)-1}r_t,

\qquad

\widehat J_{e,S}(p)=\frac{1}{|S|}\sum_{s\in S}R_e(p,s),

\]

  

where \(s\) is an episode seed and \(T\) is determined by termination or the registered episode limit. Higher reward is better, including on tasks with negative returns. Wall-clock timeout is a failure, not an ordinary truncated episode or a return of zero.

  

The search panel is \(S=\{0,\ldots,9\}\). Programs enter the elite table only after valid evaluation on this common panel. The report panel is \(H=\{1000,\ldots,1099\}\), disjoint from both the search seeds and the inspected pilot panel, which used seeds 100–199. The objective used for selection is \(\widehat J_{e,S}\); \(\widehat J_{e,H}\) is reserved for reporting. Neither the main algorithm nor its winner selection may consume held-out outcomes. These definitions follow the [prospective protocol](../companion/examples/elitelist_papers/PROTOCOL.md).

  

The program interface is `Solution(Policy)`, with asynchronous `act` and optional episode initialization through `reset`. The policy receives observation and action spaces and environment instructions, but no reward stream, environment object, or `info`/action-mask channel. The prompt requires episode state to be reset, actions to be finite and valid for their space, and any random choices to use the supplied seeded generator. The language model sees task documentation and measured search feedback between proposals; it does not choose actions during an episode. [Policy contract](../companion/elitesearch/prompts/context.j2); [runner](../companion/examples/elitelist_papers/run.py).

  

## 3. Related work

  

FunSearch evolves programs using a pretrained, frozen language model and an evaluator. Its reported design includes a program skeleton, feedback from previously evaluated programs, and an island-based program database. This establishes a direct precedent for using execution to select language-model-generated programs. EliteTable's implementation instead uses one retained elite table with generation barriers and a specified mixture of proposal operators. This is a description of the present implementation, not evidence that this organization is better. [FunSearch author manuscript](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf).

  

AlphaEvolve describes an evolutionary coding agent that proposes code changes and receives feedback from one or more evaluators, with applications to scientific and computational problems. It provides a broader precedent for iterative code modification under external evaluation. The present study applies an explicit, smaller protocol to control-policy programs and does not replicate AlphaEvolve's reported experiments or compare against its performance. [AlphaEvolve](https://arxiv.org/abs/2506.13131).

  

Gymnasium supplies the task interface and installed environments. CarRacing exposes a 96×96 RGB observation and, in the selected continuous-action profile, steering, throttle, and braking controls. These observations permit generated policies to implement their own image processing within their source code. The study adds no hand-designed observation transformation outside the policy. Live documentation is useful context; archived installed settings and source identify the actual experiment. [Official CarRacing documentation](https://gymnasium.farama.org/environments/box2d/car_racing/); [environment construction](../companion/examples/elitelist_papers/run.py).

  

These sources situate the proposal–evaluation mechanism. They do not constitute an exhaustive survey of programmatic control, nor establish priority for any individual operator or application described here.

  

## 4. Method

  

### 4.1. Elite table and generation barrier

  

EliteTable is implemented by `EliteSearch`. Each candidate record stores an organism ID, generation, operator, parent IDs, source, description, policy ID, status, per-seed rewards, repair history, and model-call records. The elite table contains at most ten successfully measured candidates retained across generations. Candidate generation and evaluation overlap within a generation; all candidates use the previous generation's elites. Promotion occurs only after that generation's candidates have finished evaluation or exhausted their repair allowance. [Agent](../companion/elitesearch/agent.py); [records](../companion/elitesearch/records.py).

  

Generation one starts with an empty table and fifty fresh proposal slots. With at least two elites, a later generation allocates ten fresh slots, twenty remix slots, and twenty edit slots. Operator slots are shuffled using the search RNG before candidate IDs and parents are assigned. Edit parents are sampled uniformly from the current elites. Each remix samples up to three distinct elites uniformly without replacement; different candidate slots may reuse parents. With exactly one elite, remixing is unavailable and its slots become edits. With no elites, all fifty slots are fresh again. Thus the ordinary allocation is 20% fresh, 40% remix, and 40% edit, with explicit fallback behavior for an unfilled table. [Population allocation](../companion/elitesearch/agent.py).

  

### 4.2. Proposal operators and feedback

  

Fresh proposals return complete modules. Their prompt asks for an approach distinct from the current elites and includes elite names, descriptions, and measured means, but not their source code. Consequently, “fresh” describes the operator; later fresh proposals are not independent of search history. Edits return exact search/replacement pairs applied sequentially to one parent's source. Each search segment must match exactly once. Remixes return complete source using up to three parent programs, with a description of the intended inheritance. Edit and remix prompts include parent source, mean reward, and per-seed search returns. [Fresh prompt](../companion/elitesearch/prompts/new.j2); [edit prompt](../companion/elitesearch/prompts/edit.j2); [remix prompt](../companion/elitesearch/prompts/remix.j2).

  

All operators receive the environment contract and the list of available worker libraries. Instructions are extracted from the installed environment documentation and resolved settings. For BipedalWalker, the runner also supplies explicit observation scaling, actuator semantics, reward details, and termination information. These are task-specific human-supplied inputs that must accompany the prompts in the artifact; the study does not test learning control from an unspecified interface. [Runner](../companion/examples/elitelist_papers/run.py).

  

The selected model identifier is `openai/gpt-oss-120b:nitro`, called through the recorded OpenRouter provider. No model-weight updates are performed. The archive records requested parameters and returned model/provider metadata because a stable identifier does not ensure a stable hosted implementation or deterministic sampling. The current runner sets a maximum output length of 16,384 tokens and a 120-second provider timeout; the search agent also applies its 120-second model-call timeout. [Study configuration](../companion/study.json); [runner](../companion/examples/elitelist_papers/run.py); [agent](../companion/elitesearch/agent.py).

  

### 4.3. Validation, deduplication, and repair

  

Generated outputs must satisfy their structured schema. Static checks parse source without importing it, require a top-level `Solution` class, and check supported constructor signatures and edit boundaries. The prompts request inherited construction and forbid introducing evolution markers; the shared edit machinery nevertheless preserves any marker-delimited immutable material if present. Without markers, the entire source is editable. Static validation is not a proof of correct policy behavior. [Validation and editing](../companion/alphaevolve/edits.py).

  

Within a search, a candidate is rejected when its parsed Python AST matches a previously accepted source AST after source-location attributes are removed. The set also retains accepted intermediate sources that later fail execution and undergo repair. This catches some cosmetic duplication, such as formatting-only changes, but does not establish behavioral uniqueness. Renamed or structurally different programs may still implement the same controller. [Agent](../companion/elitesearch/agent.py).

  

Malformed generated outputs, invalid edits, duplicate ASTs, and candidate execution failures enter a shared repair loop. A repair receives the failed material, diagnostic, and original reference for an edit, then returns complete replacement source. Each candidate slot permits at most five repair calls in total, including repairs following execution failures. A slot is discarded if its allowance is exhausted. Repair consumes additional calls and can trigger additional episodes; fifty slots do not mean fifty model calls or fifty executed sources. Infrastructure and uncaught provider failures interrupt the run rather than receiving a fabricated candidate score. [Repair prompt](../companion/elitesearch/prompts/repair.j2); [agent](../companion/elitesearch/agent.py); [measurement adapter](../companion/examples/elitesearch.py).

  

### 4.4. Evaluation and retention

  

Candidate programs execute in the existing separate-process Docker backend. The container disables networking, uses a read-only root filesystem, drops capabilities, and applies resource limits. The current main configuration permits four episode workers, with ten seconds per policy call and ten seconds per episode result. The episode limit includes policy initialization and simulation, but excludes Docker startup, queueing, and final process cleanup. These are execution constraints, not a claim that every future generated policy will fit them. [Docker backend](../companion/rsikit/sandbox/docker.py); [runner contract](../companion/examples/elitelist_papers/run.py).

  

A successful candidate's score is the arithmetic mean of its ten search returns. A candidate failure is represented separately from rewards; the measurement adapter returns a failure rather than a partial-panel mean. At promotion, the previous elites and newly successful candidates are sorted by decreasing search mean. Exact ties prefer current-generation edits, then current-generation remixes, then incumbents, then current-generation fresh candidates; organism ID ascending breaks the remaining ties. The first ten form the new table. This order is part of the method and can change the selected policy even without a search-score increase. [Ranking implementation](../companion/elitesearch/agent.py).

  

Because incumbents remain eligible, the best retained search score cannot decrease across completed generations with a valid incumbent. This is a property of the retention rule, conditional on the stored measurements; it is not evidence of held-out improvement. In particular, a tie replacement or a candidate favored by the finite search panel can reduce held-out reward.

  

### 4.5. Search pseudocode

  

```text

Inputs: environment e; fixed LLM; search seed q; search panel S

E = empty elite table; A = empty set of accepted source ASTs

for generation g = 1,...,10:

allocate and shuffle 50 operator slots from E

sample each slot's parents from the unchanged E

concurrently, subject to model and episode worker limits:

propose each candidate with its assigned operator

validate output, source, edit constraints, and AST uniqueness against A

evaluate valid source on S, adding accepted source ASTs to A

on candidate failure, repair and repeat, at most 5 repairs per slot

record successful search returns or final discard

wait for every slot to finish; abort/checkpoint infrastructure interruptions

E = top 10 of E plus successfully measured candidates

ordered by (-mean search reward, operator/incumbent tie priority, ID)

save ordered elite IDs, lineage, calls, failures, and generation status

after search:

evaluate each distinct recorded generation winner and random reference on H

report held-out returns without changing winners or repairing on H

```

  

**Algorithm 1.** Search and reporting are separate phases. Allocation uses the fallback rules in Section 4.1; the generation barrier fixes the parent table during each generation. The actual implementation persists intermediate state as well as the completed-generation checkpoint.

  

### 4.6. Persistence and resumption

  

Checkpoints preserve candidates, revisions, model calls, generation records, and cached episode evidence. Resume reconstructs the allocation from the search seed and verifies its consistency with the recorded parents and configuration. Existing evaluated or discarded candidates are retained, and available policy/seed evaluations are reused. Model calls can be nondeterministic; recovery preserves recorded work without claiming that an interrupted request can be reproduced bit for bit. Attempt histories and source snapshots document restarts. A main replicate remains one search across attempts rather than becoming a new independent replicate each time it resumes. [Restore and continuation](../companion/elitesearch/agent.py); [run persistence](../companion/rsikit/run.py); [attempt handling](../companion/examples/elitelist_papers/run.py).

  

## 5. Prospective experiment

  

### 5.1. Task suite and budget

  

The agreed suite uses installed Gymnasium 1.3.0 profiles with native observations and registered episode limits. The following limits and constructor choices are recorded in the existing environment manifests and are expected for the main configuration; every main manifest must record its resolved values. The [runner](../companion/examples/elitelist_papers/run.py) and [study configuration](../companion/study.json), rather than an illustrative documentation example, specify the launch behavior.

  

| Environment | Profile | Episode step cap | Reporting target |

| --- | --- | ---: | ---: |

| CartPole-v1 | Standard reward (`sutton_barto_reward=false`) | 500 | 475 |

| MountainCar-v0 | Discrete actions; goal velocity 0 | 200 | −110 |

| Acrobot-v1 | Standard installed profile | 500 | −100 |

| LunarLander-v3 | Discrete actions; gravity −10; wind disabled | 1,000 | 200 |

| BipedalWalker-v3 | Non-hardcore terrain | 1,600 | 300 |

| CarRacing-v3 | Continuous actions; domain randomization disabled; lap fraction 0.95 | 1,000 | 900 |

  

**Table 1.** Planned environment profiles and descriptive targets. Settings are supported by the archived [pilot manifests](../companion/data/pilots/) and must be confirmed in main-run manifests. Targets annotate outcomes and do not stop search.

  

Each environment has ten separately initialized searches, with search seeds 0–9. All use the same ten search episode seeds and one hundred held-out episode seeds. The search seed controls operator ordering and parent selection; it does not make hosted LLM responses deterministic. Replicates have separate elite tables and histories, although they share the environment panels and model service.

  

Every complete search schedules ten generations of fifty candidate slots, retaining ten elites. This yields 500 scheduled slots per search and 30,000 across sixty complete searches. At ten search episodes per slot, the nominal count is 300,000 candidate-episode evaluations before repairs, failures, and cache reuse. These arithmetic planning counts are not measured steps, runtime, cost, or guarantees about the number of successful policies. Repair calls and held-out evaluations require separate accounting. [Protocol](../companion/examples/elitelist_papers/PROTOCOL.md).

  

All new main runs use `--no-early-stop`; the stopping target is null while a separate `reporting_target` retains the task threshold. Previously early-stopped pilots cannot be reclassified as main replicates. Searches begin in new `main-v1-*` directories. The initial launch plan runs searches sequentially, with fifty concurrent model requests and four episode workers within a search. [Command generator](../companion/tools/study.py).

  

### 5.2. CarRacing case study and selection disclosure

  

CarRacing is featured because inspection of its exploratory pilot trajectory motivated interest in continued improvement. Its choice is therefore pilot-driven, not a task selected without seeing outcomes. The main analysis must retain all ten CarRacing replicates regardless of whether they reproduce that trajectory. The pilot [curve file](../companion/data/pilots/pilot-CarRacing-v3-0/curves.csv), [held-out evidence](../companion/data/pilots/pilot-CarRacing-v3-0/heldout.json), and [resume history](../companion/data/pilots/pilot-CarRacing-v3-0/resumes/) remain separate exploratory evidence.

  

The controller sees the native 96×96×3 image, including the rendered indicators that are part of that observation. There is no external state-vector replacement. Any image features, filtering, or memory are implemented by the generated policy. This makes the source at generations one, five, and ten useful descriptive material, but source inspection alone cannot determine which mechanism caused a reward change. [Official observation specification](https://gymnasium.farama.org/environments/box2d/car_racing/).

  

### 5.3. Held-out reporting and failures

  

After search, the runner identifies the first elite at each completed generation from the stored search ranking. Each distinct selected policy is evaluated on the held-out panel; repeated winners share their cached evidence. The random-action policy is also evaluated on this panel. The final winner may already have been evaluated by the search wrapper, in which case the report reuses available evidence. No held-out result changes the checkpoint winner, enters a repair prompt, or influences further main-search proposals. [Historical-winner export](../companion/examples/elitelist_papers/run.py).

  

The evidence inventory includes all sixty planned searches, including missing directories, incomplete generations, invalid candidates, timeouts, and failed or incomplete held-out panels. Analysis eligibility requires completed run status, exactly the ten registered search-generation checkpoints, and agreement with the frozen protocol, including recorded attempts. A primary endpoint additionally requires valid generation-one and generation-ten panels, each covering all one hundred registered held-out seeds, with finite returns and no evaluation failure. Missing intermediate held-out panels do not exclude an otherwise eligible endpoint pair. Missing rewards remain missing. An interrupted search may resume with the same protocol and cached work; an irrecoverable search stays in the inventory. Any replacement is an additional labeled search, never an invisible substitution for the original. Scientific protocol changes require a documented amendment and a separate group. [Protocol](../companion/examples/elitelist_papers/PROTOCOL.md); [analysis eligibility](../companion/tools/results.py).

  

## 6. Prespecified analysis

  

### 6.1. Endpoint and uncertainty

  

For environment \(e\), search \(q\), and generation \(g\), let \(p_{e,q,g}\) be the search-selected winner and define

  

\[

Y_{e,q,g}=\frac{1}{100}\sum_{s=1000}^{1099}R_e(p_{e,q,g},s),

\qquad

\Delta_{e,q}=Y_{e,q,10}-Y_{e,q,1}.

\]

  

The primary summary is \(m_e=\operatorname{median}_q\Delta_{e,q}\) over valid paired searches, reported with the eligible count and all sixty inventory statuses. The endpoint is generation ten minus generation one; it is not the best held-out generation minus the first, nor the last available generation of an incomplete run.

  

For each environment, a descriptive 95% percentile bootstrap interval uses 10,000 resamples of whole paired search IDs with replacement and RNG seed 20260925. Each resample retains the endpoint pair from a search and recomputes the median difference. The interval is the 2.5th and 97.5th percentiles of those bootstrap medians. Episodes are not treated as independent search replicates. With fewer than two valid pairs the interval is undefined; with no valid pairs the median is also undefined. With only ten planned searches per task, intervals have limited precision and do not establish population-wide reliability. Conditioning the endpoint summary on available pairs also makes failure reporting essential.

  

### 6.2. Curves, targets, and references

  

Per-environment plots will show all ten generation checkpoints, individual valid search trajectories, the median, the interquartile range, and valid-run counts. The interquartile range describes observed dispersion and is not a confidence interval. A curve summary may use different available searches at different generations; the paired endpoint will explicitly identify its fixed eligible set.

  

Search-panel target attainment means reaching the target at any recorded generation; the held-out target summary uses generation ten. They will be reported separately, including their denominators. Generation-one attainment exposes cases where the task is already saturated for the selected policy. Changes after first search-target attainment will be descriptive secondary summaries and will not replace the prespecified endpoint. Generation one is a selected result from a fifty-slot search, not a single unconditioned model sample. The random-action reference is an orientation point, not a competitive or budget-matched comparator. Raw reward differences are reported separately by task; no cross-task ranking of normalized effect is implied.

  

### 6.3. Representative policies

  

For each environment, select among eligible runs with valid held-out evidence at all ten generations the run closest to the median paired held-out improvement, breaking distance ties with the lower search seed:

  

\[

q_e^*=\arg\min_q\bigl(|\Delta_{e,q}-m_e|,q\bigr),

\]

  

with lexicographic ordering. The reference median \(m_e\) uses all eligible endpoint pairs, including pairs with missing intermediate held-out panels; only the illustration candidates require all ten panels. Missing source checkpoints are disclosed explicitly. This held-out-based choice is solely for illustration after analysis and cannot alter the primary result or search. If no run meets the illustration requirements, the representative is undefined. [Selection implementation](../companion/tools/results.py).

  

The CarRacing illustration will include source and parent lineage for generations one, five, and ten from this representative run. It will identify persistent mechanisms, revised parameters, and changes in source where supported by the actual code. A successful descendant does not prove that a named operator or inherited component caused improvement. No main representative can be selected before main data exist.

  

### 6.4. Resource accounting

  

For every search and across its attempts, report scheduled slots, successful candidates, discarded slots, repair calls, total recorded calls, returned token usage and cost, and elapsed time. Invalid proposals and failed evaluations remain visible even if later repaired. Missing provider usage or billing information remains unknown. Concurrent request durations must not be summed and labeled wall-clock run time, and repeated resume snapshots must not be counted as new attempts or charges. No exact environment-step efficiency claim will be made without recorded and validated step counts. [Call logging and attempt records](../companion/examples/elitelist_papers/run.py).

  

## 7. Results: pending main experiment

  

**There are no main-study results in this draft.** The following is a planned table schema, with all outcome cells intentionally empty. Blank cells indicate pending measurements, not zero reward, failure, or a measured absence of improvement. Pilot values will not fill these cells.

  

| Environment | Planned searches | Complete searches | Valid endpoint pairs | Median generation-1 held-out mean | Median generation-10 held-out mean | Median paired change | 95% bootstrap interval |

| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |

| CartPole-v1 | 10 | | | | | | |

| MountainCar-v0 | 10 | | | | | | |

| Acrobot-v1 | 10 | | | | | | |

| LunarLander-v3 | 10 | | | | | | |

| BipedalWalker-v3 | 10 | | | | | | |

| CarRacing-v3 | 10 | | | | | | |

  

**Table 2.** Planned primary results. The median paired change is computed from within-search differences and need not equal the difference of the two marginal medians. Counts and missing-evidence reasons will accompany every populated row.

  

The completed results section will additionally contain the per-generation figures specified in Section 6, a separate search/held-out target table, the full run inventory, representative-policy examples, and resource accounting. Until those artifacts exist, no conclusion about improvement, saturation, reliability, or relative performance is supported by the main study.

  

The available execution preflight addresses a different question: whether selected archived policies and the reporting workflow can run in the prospective companion environment. Its report documents replay checks, an interruption/resume exercise using scripted proposals, and offline evidence regeneration. These checks support engineering readiness for the tested cases; they do not estimate EliteTable's algorithmic efficacy or replace the sixty planned searches. The [preflight report](../companion/PREFLIGHT.md), [raw preflight summary](../companion/preflight/summary.json), and [scripted-resume artifacts](../companion/preflight/scripted-resume/) provide the evidence separately. This manuscript introduces no new replay or search measurements.

  

## 8. Reproducibility and artifact status

  

The companion includes source, prompts, environment settings, dependency records, exploratory archives, and tools for offline analysis, saved-policy replay, and command generation. These support three distinct checks: regenerating numeric evidence from archived returns without API access; reevaluating archived source under the recorded environment; and launching a new search under the same protocol. Main-study analysis uses `python -m tools.analyze --main` from the companion root; it applies the frozen eligibility rules and retains the entire planned inventory. Only offline regeneration is intended to reproduce numeric tables exactly. A new hosted-model search can produce different source and outcomes. [Companion instructions](../companion/README.md).

  

The prospective worker is a Python 3.14.7 Linux ARM64 image identified in [study.json](../companion/study.json) by `sha256:c5781ed526fbccfb3124e6fa1b913f44b3fd1d91436b108494dd8f8065d1a010`. The archived image and its file checksum are recorded in [ARTIFACTS.json](../companion/ARTIFACTS.json), while [worker-environment.json](../companion/worker-environment.json) records worker dependencies and source. Rebuilding a Dockerfile with dependency ranges is not guaranteed to recover that exact image. Other architectures or emulation require separate timing validation for the same wall-clock limits.

  

Source snapshots identify actual bytes, including uncommitted development changes that a parent Git revision alone cannot capture. [SOURCE_PROVENANCE.json](../companion/SOURCE_PROVENANCE.json) records copied-source origins, and [CHECKSUMS.json](../companion/CHECKSUMS.json) records packaged file hashes. The frozen main-study protocol is prepared, and the execution preflight has passed. The [release record](../companion/FREEZE.md) and packaged hashes identify the study configuration, prompts, tie order, dependencies, and image to preserve at launch. Publication licensing and third-party notice checks remain release work; the companion is not represented here as a published repository. [Artifact status](../companion/PREFLIGHT.md).

  

## 9. Limitations

  

**Model prior knowledge and task familiarity.** The environment names, documentation, and explicit control details are provided to a pretrained model. These familiar tasks may admit remembered heuristics or recombinations of known controllers. This protocol cannot distinguish discovery from retrieval, demonstrate learning from scratch, or establish generality to unfamiliar environments.

  

**Finite evaluation panels.** Reusing ten search seeds encourages adaptation to those particular episodes. The separate one-hundred-seed panel measures transfer within the same environment profile, not robustness to new physics, unseen observation conventions, or distribution shifts. Shared panels across searches also mean that the bootstrap describes variation across search histories conditional on those panels.

  

**Small replicate count and missingness.** Ten searches per task permit descriptive comparisons but limited precision. Failed or missing endpoint panels can make valid-pair summaries unrepresentative of all planned attempts. Keeping those attempts visible prevents silent exclusion; it does not remove the inferential limitation.

  

**Pilot-driven choices.** Task emphasis and execution decisions were informed by exploratory runs. CarRacing's featured status must remain disclosed, and disappointing main trajectories must receive the same reporting treatment as favorable ones. Pilot and main seed separation does not erase the role of pilot-guided study design.

  

**Execution and provider dependence.** Wall-clock limits make eligibility partly hardware-dependent. The replay preflight covers selected policies on one workstation, not every source that the model may generate. Provider routing, model-service changes, concurrency, and interrupted requests can also affect reproducibility despite fixed user-facing settings and archived response metadata.

  

**No comparative or causal identification.** A positive generation-ten-minus-one difference would describe the combined search procedure. Without a budget-matched independent-generation condition, it would not establish that evolutionary reuse outperforms drawing additional programs. Without ablation, it would not identify the value of remixing, repairs, tie preferences, or any particular mechanism in a representative policy. Code accessibility alone is not a measured interpretability result.

  

## 10. Conclusion

  

EliteTable specifies an executable-policy search procedure with fixed model weights, measured elite retention, explicit proposal operators, bounded repair, and persistent evidence. The prospective experiment separates search selection from held-out reporting and preserves the full planned search inventory. The main empirical question remains open until the sixty searches and their reporting panels are collected. The eventual conclusion must be limited to those observed trajectories, failures, and resource measurements.

  

## Appendix A. Implementation and evidence map

  

| Item | Source or required artifact | Verification target |

| --- | --- | --- |

| Population, parents, repairs, ranking | [agent.py](../companion/elitesearch/agent.py) | Match Section 4, including underfilled elites and exact tie order |

| Exact prompt text | [context](../companion/elitesearch/prompts/context.j2), [new](../companion/elitesearch/prompts/new.j2), [edit](../companion/elitesearch/prompts/edit.j2), [remix](../companion/elitesearch/prompts/remix.j2), [repair](../companion/elitesearch/prompts/repair.j2) | Archive complete templates, worker-library text, and resolved context |

| Seed panels and execution arguments | [study.json](../companion/study.json), per-run `experiment.json` | Match main configuration and preserve reporting target separately |

| Search winners and lineage | Per-run `run.sqlite`, generation elite IDs, candidate parent IDs | Select on search ranking without executing source during analysis |

| Endpoint and curves | Per-run `heldout.json`, `curves.csv`, checkpoint policy IDs | Require complete registered panels and recompute means from raw returns |

| Inventory and failures | Planned study grid, `status.json`, `attempts.json`, candidate records | Retain all sixty searches and label extra replacement runs |

| Resource totals | `llm_calls.jsonl`, candidate calls/repairs, attempt timing | Avoid resume duplication; distinguish unknown billing from zero |

| Representative policy | Eligible paired endpoint table, complete held-out trajectories, and archived source | Use the median of all eligible pairs; restrict illustration candidates to all ten panels; break distance ties by lower search seed |

| Reproduction | [README](../companion/README.md), provenance, checksums, worker archive | Separate offline regeneration, policy replay, and new searches |

  

**Table A1.** Claim-to-artifact map. Main-run paths are evidence requirements, not a statement that the files already exist.

  

## Appendix B. Planned outputs and completion conditions

  

The manuscript can be completed from the main archive when each planned search has an explicit inventory status, all available checkpoints and held-out panels are audited, and the primary table and curves are regenerated from raw returns. The results must state eligible denominators, incomplete-panel reasons, target definitions, and unknown resource quantities. Source examples must follow the predetermined representative rule. Every outcome sentence must point to a generated table or archived measurement rather than to the pilot narrative or execution preflight.

  

Before publication, record the frozen release identifier, any protocol amendments, actual provider metadata, hardware context, exact environment settings, and licensing decisions. No illustrative numeric result should be substituted while any of these evidence steps remains unfinished.

  

## Sources

  

1. Bernardino Romera-Paredes et al. *Mathematical discoveries from program search with large language models.* Nature 625, 468–475 (2024; published online December 2023). [Publisher article](https://www.nature.com/articles/s41586-023-06924-6). [Primary author manuscript](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf).

2. Alexander Novikov et al. *AlphaEvolve: A coding agent for scientific and algorithmic discovery.* arXiv:2506.13131 (2025). [Primary paper](https://arxiv.org/abs/2506.13131).

3. Farama Foundation. *Car Racing — Gymnasium documentation.* [Official environment documentation](https://gymnasium.farama.org/environments/box2d/car_racing/). Consulted 25 September 2026; archived installed configuration takes precedence for this experiment.

4. EliteTable companion. [Prospective protocol](../companion/examples/elitelist_papers/PROTOCOL.md), [study configuration](../companion/study.json), [implementation](../companion/elitesearch/agent.py), [experiment runner](../companion/examples/elitelist_papers/run.py), [analysis](../companion/tools/results.py), and [preflight record](../companion/PREFLIGHT.md). Local primary implementation and preparation evidence, dated 25 September 2026; packaged hashes identify the main-study launch inputs.
