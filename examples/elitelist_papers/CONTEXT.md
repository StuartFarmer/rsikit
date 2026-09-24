> In independent research, is it better to publish many smaller papers that build upon each other or one large one

Usually, publish a sequence of smaller, coherent papers that build toward a larger synthesis.

Each paper should contain a complete, defensible contribution—not merely a fragment. This approach gives you:

* Faster feedback and earlier evidence of credibility
* Multiple opportunities for citation and discovery
* Less risk that years of work culminate in one rejected manuscript
* A public record showing how the research program developed
* Freedom to revise later stages as results emerge

A strong structure is:

1. **Framework paper** — define the question, terminology, model, or dataset.
2. **Methods paper** — explain and validate the research or simulation method.
3. **Results papers** — address distinct hypotheses or experiments.
4. **Synthesis paper** — combine the findings and explain their broader significance.

One large paper is preferable when the components are not independently meaningful, later sections are necessary to validate earlier claims, or dividing it would create “salami slicing”—several thin publications from what is substantively one result.

For your economic-game and penny-auction research, I’d likely separate:

* A general agent-based framework for auction games
* Validation against known Dollar Auction results
* Strategy and equilibrium analysis
* Extension to penny auctions
* Tokenized/on-chain implementation and mechanism-security analysis
* A final synthesis comparing mechanisms

That gives you an identifiable research program rather than one enormous paper—or a pile of disconnected small ones. The practical rule is: **one paper per independently useful claim, not one paper per experiment.**

> I have my own automated LLM problem solver like alphaevolve. I want a simple paper explaining the model and then follow ups on performance in different environments, comparison to other models etc

Yes—this is an ideal case for a foundational paper followed by focused empirical papers.

## Paper 1: The system paper

Publish a concise but complete description of the automated problem solver:

**Possible title:**
*An Automated LLM-Guided Framework for Iterative Problem Solving*

Include:

* The problem the system addresses
* System architecture and optimization loop
* How candidates are generated, evaluated, selected, and revised
* Roles of the underlying LLM and external evaluator
* Pseudocode sufficient to reproduce the algorithm
* Stopping conditions and compute budget
* A small set of representative experiments
* Ablations demonstrating which components matter
* Limitations, failure modes, and safety considerations
* Code and evaluation protocol, if you can release them

This paper establishes the named method. It still needs credible baseline results; an architecture-only paper will usually be treated more like a proposal or technical report.

## Follow-up papers

Each subsequent paper should answer a distinct research question.

1. **Broad benchmark evaluation**
   How well does the system generalize across optimization, mathematics, algorithms, and software engineering?

2. **Baseline comparison**
   Compare against AlphaEvolve-style methods, evolutionary search, best-of-\(N\), self-refinement, agent scaffolds, and direct LLM generation under matched inference budgets.

3. **Domain-specific studies**
   Use separate papers only where the environment requires meaningful methodological changes or produces substantial domain insight—for example:

   * Auction-mechanism discovery
   * Quantitative strategy optimization
   * Smart-contract engineering
   * Scientific or mathematical discovery

4. **Scaling and efficiency**
   Study performance as a function of model size, number of generations, population size, evaluator calls, tokens, wall-clock time, and monetary cost.

5. **Mechanistic and ablation analysis**
   Investigate diversity, mutation operators, selection pressure, archive design, evaluator reliability, and which components actually cause improvement.

## Make the sequence scientifically credible

Freeze and timestamp the core algorithm before running the follow-up evaluations. Otherwise, readers may reasonably suspect that each environment influenced the system’s design.

For every comparison, match or report:

* Total tokens and LLM calls
* Evaluator or execution calls
* Compute and wall-clock time
* Number of independent runs
* Model versions and prompts
* Search-space access
* Statistical uncertainty
* Failed and invalid candidates

The first paper should claim **“here is the method and initial evidence.”** It should not claim general superiority. Later papers can establish generalization, comparative performance, and scaling.

I would avoid publishing one paper for every benchmark. Group environments by a real scientific thesis. The dividing rule remains: **each paper should resolve a distinct question, not merely report another score.**

> What are the representative experiments

For the foundational paper, representative experiments should demonstrate the system’s core capabilities without trying to prove universal superiority. I’d use four environments with objective, automatically verifiable scores.

| Experiment                        | Example task                                                    | What it demonstrates                              |
| --------------------------------- | --------------------------------------------------------------- | ------------------------------------------------- |
| Algorithm synthesis               | Generate an efficient sorting, packing, or graph algorithm      | Produces correct executable solutions             |
| Code optimization                 | Optimize a slow reference implementation while preserving tests | Iteratively improves an existing artifact         |
| Combinatorial optimization        | Bin packing, traveling salesperson, scheduling, or knapsack     | Searches a measurable fitness landscape           |
| Scientific/mathematical discovery | Discover a formula, heuristic, bound, or numerical method       | Produces novel candidates rather than merely code |

## 1. Algorithm synthesis

Give the system a specification, test suite, and performance objective.

For example:

> Implement an algorithm for dynamic weighted interval scheduling. Correctness is mandatory; minimize runtime across hidden instances.

Measure:

* Percentage of valid candidates
* Pass rate on hidden tests
* Best runtime achieved
* Evaluations required to reach the best solution
* Variance across repeated runs

This establishes that the system can generate and improve executable solutions under hard correctness constraints.

## 2. Existing-code optimization

Start from a correct but deliberately inefficient implementation. Ask the system to improve it without changing observable behavior.

Good targets include:

* Matrix or tensor operations
* Data compression/decompression
* Parsing or serialization
* Database query execution
* A numerical kernel
* One of your order-book processing routines

Measure:

$$
\text{speedup}=\frac{\text{baseline runtime}}{\text{optimized runtime}}
$$

Also report memory usage, correctness, code size, compilation failures, and whether the improvement generalizes to unseen inputs.

This is especially representative of an AlphaEvolve-like system because it shows iterative artifact evolution rather than one-shot code generation.

## 3. Combinatorial optimization

Choose a problem with a known objective and established solvers:

* Bin packing
* Traveling salesperson
* Job-shop scheduling
* Maximum independent set
* Vehicle routing

The most interesting setup is to ask the model to evolve a reusable heuristic—not directly solve one fixed instance. Train or search on one collection of instances, then evaluate the discovered heuristic on unseen instances.

Measure:

$$
\text{optimality gap}
=
\frac{\text{candidate score}-\text{best-known score}}
{\lvert\text{best-known score}\rvert}
$$

This tests whether the system discovers general strategies instead of memorizing individual answers.

## 4. Open-ended discovery

Give the system a compact domain where discoveries are easy to verify. For example:

* Find a better approximation formula
* Discover a cellular-automaton rule with a desired property
* Construct a high-performing auction strategy
* Discover a numerical integration heuristic
* Improve a graph invariant or construction
* Design a heuristic for the Dollar Auction or penny-auction environment

This showcases the system’s most ambitious capability: searching for artifacts that were not explicitly provided in its prompt.

Keep this experiment modest in the first paper. A convincing, interpretable discovery is more valuable than a dramatic but difficult-to-validate claim.

## Essential comparisons

For every environment, compare against:

1. The original human-written baseline
2. A single direct LLM response
3. Best-of-\(N\) independent LLM samples
4. Iterative self-refinement without population or selection
5. Random or conventional evolutionary search, where applicable
6. Your complete system

The critical comparison is with **best-of-\(N\) under the same inference budget**. Otherwise, any improvement may simply come from generating more candidates.

## Essential ablations

Disable one component at a time:

* No candidate archive or memory
* No evolutionary history in context
* No crossover
* No mutation diversification
* Random rather than fitness-based selection
* Single parent rather than population search
* No evaluator feedback
* No LLM-generated analysis before modification

This answers the important question: **is the architecture producing the improvement, or is it merely repeated sampling?**

## Recommended minimum package

For a simple first paper, I would run:

* **Three environments:** code optimization, combinatorial heuristic discovery, and one open-ended discovery task
* **Five tasks per environment**
* **At least 10 independent runs per task and method**
* **Equal token/evaluation budgets**
* **Held-out test instances**
* **Four primary baselines**
* **Ablations on the three most distinctive system components**

The paper should show three representative trajectories in detail:

1. A clear success showing successive improvements
2. A typical run showing ordinary progress
3. A failure showing stagnation, evaluator exploitation, or invalid solutions

The individual task scores matter, but the central experiment is whether your solver repeatedly produces improvement curves like:

$$
S_0 < S_1 < S_2 < \cdots < S_{\text{best}}
$$

under controlled budgets, and whether those gains survive held-out evaluation. That would support the foundational claim that the system is a genuine automated improvement process.

> How do I do ablation on this

Ablation means systematically removing or weakening one part of your solver and measuring how much performance changes. You’re trying to establish which components cause the gains.

Start by defining the full system as \(F\), then create controlled variants:

| Variant              | Change                                          | Question answered                    |
| -------------------- | ----------------------------------------------- | ------------------------------------ |
| Full                 | Everything enabled                              | Reference performance                |
| No archive           | Remove persistent candidate/history storage     | Does accumulated knowledge help?     |
| No feedback          | Don’t show evaluator feedback to the LLM        | Is feedback driving improvement?     |
| No selection         | Select parents randomly                         | Does fitness-based selection matter? |
| No population        | Keep only the current best candidate            | Does population diversity matter?    |
| No crossover         | Generate from one parent only                   | Does combining candidates help?      |
| No mutation strategy | Use identical generic improvement prompts       | Do specialized operators matter?     |
| No reflection        | Remove LLM analysis before generation           | Does explicit reasoning help?        |
| No iteration         | Generate the same number of independent samples | Does evolution beat best-of-\(N\)?   |

Not all of these will apply to your architecture. Ablate only components you actually claim are important.

## The basic experimental procedure

For every task:

1. Fix a total budget, such as:

   * 100 candidate evaluations
   * 200,000 generated tokens
   * 30 minutes of execution
   * Or a fixed monetary cost

2. Run the full system and every ablation with:

   * The same underlying LLM
   * Same model parameters
   * Same initial candidate
   * Same task instances
   * Same evaluator
   * Same stopping rules
   * Same total resource budget

3. Perform multiple independent runs using predetermined random seeds.

4. Evaluate final candidates on held-out instances.

5. Compare both final performance and search behavior.

The main effect of component \(c\) is:

$$
\Delta_c = \operatorname{Score}(F)-\operatorname{Score}(F\setminus c)
$$

If higher scores are better, a positive \(\Delta_c\) suggests that component contributes value.

## Example: ablating evaluator feedback

Suppose your full loop is:

1. Select a candidate.
2. Modify it with the LLM.
3. Execute and score it.
4. Feed the score and error information back to the LLM.
5. Store successful candidates.

Create three conditions:

* **Full feedback:** score, errors, test failures, and performance profile
* **Score only:** numerical fitness without diagnostics
* **No feedback:** LLM knows only that it should produce another candidate

If the results are:

| Condition     | Median final score | Success rate |
| ------------- | -----------------: | -----------: |
| Full feedback |                 92 |          80% |
| Score only    |                 84 |          57% |
| No feedback   |                 69 |          23% |

you can argue that evaluator feedback matters—and that descriptive diagnostics add value beyond the fitness number alone.

## Most important control: best-of-\(N\)

This is effectively the “everything removed” ablation.

Generate \(N\) independent solutions from the same LLM and return the highest-scoring one. Give best-of-\(N\) the same number of LLM generations and evaluator calls as the complete solver.

Compare:

```text
Full solver:
candidate → evaluation → feedback → improved candidate → ...

Best-of-N:
candidate 1
candidate 2
candidate 3
...
candidate N
```

If the full solver does not outperform budget-matched best-of-\(N\), then its iterative architecture may not add much beyond repeated sampling.

## What to measure

Do not report only the best score. Record:

* Best final held-out score
* Median result across runs
* Success rate
* Number of evaluations to reach a target score
* Area under the best-score curve
* Invalid candidate rate
* Candidate diversity
* Evaluator exploitation or reward-hacking rate
* Generated tokens, execution time, and monetary cost

For optimization, a useful sample-efficiency metric is:

$$
\operatorname{AUC}
=
\frac{1}{B}\sum_{t=1}^{B}S_t^{\text{best}}
$$

where \(B\) is the evaluation budget and \(S_t^{\text{best}}\) is the best score found by evaluation \(t\). This distinguishes a system that improves quickly from one that happens to get lucky at the end.

## Use repeated runs, not candidate-level samples

The unit of analysis should be an entire solver run. Candidates generated within one run are dependent and should not be treated as independent observations.

A reasonable starting point is:

* 10–20 independent runs per condition
* Identical predetermined seeds across conditions when possible
* Median and interquartile range
* Bootstrap confidence intervals for the performance difference

If runs are expensive, use five exploratory runs to find obvious non-contributors, then spend the full evaluation budget on the final ablation set.

## Single versus combined ablations

Begin with one-at-a-time ablations:

$$
F,\quad F-A,\quad F-B,\quad F-C
$$

But this does not reveal interactions. Selection may only help when an archive exists, for example. For your two or three most important components, run a small factorial experiment:

| Archive | Feedback | Selection |
| ------- | -------- | --------- |
| On      | On       | On        |
| Off     | On       | On        |
| On      | Off      | On        |
| On      | On       | Off       |
| Off     | Off      | On        |
| Off     | On       | Off       |
| On      | Off      | Off       |
| Off     | Off      | Off       |

You do not need a factorial experiment for every implementation detail. Reserve it for components whose interaction is central to your claimed mechanism.

## A sensible first-paper ablation set

I would prioritize:

1. **Full system**
2. **Budget-matched best-of-\(N\)**
3. **No evaluator feedback**
4. **No archive or accumulated history**
5. **Random parent selection**
6. **Single-candidate hill climbing**
7. **Full system with a weaker/cheaper underlying LLM**

Run these across at least three different task types. That lets you distinguish among three possible explanations:

* The underlying LLM is doing everything.
* Repeated sampling is doing everything.
* Your search, feedback, and memory architecture produce additional gains.

The strongest result would be that the full system improves median held-out performance and sample efficiency across environments, while different ablations hurt different aspects of the search in interpretable ways.

