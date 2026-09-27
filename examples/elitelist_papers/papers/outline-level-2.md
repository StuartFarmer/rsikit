# EliteTable — Level 2 Paper Outline

This outline defines the paper's focus: **introducing EliteTable and demonstrating what it can accomplish**, with pilot findings clearly separated from the pending main experiment.

Companion documents: [Level 1 outline](outline.md) and [working manuscript](elitetable-policy-search.md).

## Introduction

### 1. Why is your research important?

- Designing effective algorithms and control policies usually requires substantial human expertise and repeated experimentation.
- Language models can generate executable solutions, while an external evaluator can measure their performance and guide further improvement.
- Understanding how to organize this search is an important research problem: which solutions should be retained, reused, modified, or replaced?
- EliteTable investigates how much a simple procedure built around a small collection of strong solutions can accomplish. Its outputs are executable programs that can be inspected and run without further language-model calls.

### 2. What is known about the topic?

- Systems such as FunSearch and AlphaEvolve establish the use of language models and execution feedback to search for better programs.
- These approaches share a broad loop: propose a solution, evaluate it, retain useful information, and generate further candidates.
- Implementations differ in how they store solutions, select parents, introduce variation, and allocate computation.
- EliteTable explores one specific design: a bounded elite table combined with fresh proposals, edits, and recombination.
- This paper describes and evaluates that design. Comparisons against other search algorithms and component ablations remain follow-up work.

### 3. What are your hypotheses?

- **Primary hypothesis:** Repeated proposal, evaluation, and elite reuse improve the held-out performance of generated policies within a ten-generation budget.
- The measurable prediction is a positive median paired change between generation-one and generation-ten winners, evaluated separately for each environment.
- **Secondary expectation:** Some environments will reach their performance targets in the first generation, while others will benefit from continued search or remain below target. Immediate success and later improvement are distinct outcomes.

### 4. What are your objectives?

- Describe EliteTable clearly enough for others to implement and reproduce it.
- Evaluate generated policies across six standard control environments.
- Measure performance on episodes excluded from search selection.
- Document target attainment, improvement, plateaus, failures, and resource use.
- Illustrate how selected policy programs change over successive generations.

## Materials and Methods

### 1. What materials did you use?

- The EliteTable implementation, using `EliteSearch` to manage proposals, evaluation, elite retention, and checkpoints.
- The language model `openai/gpt-oss-120b:nitro`, accessed through OpenRouter without model-weight updates.
- Six Gymnasium environments: CartPole, MountainCar, Acrobot, LunarLander, BipedalWalker, and CarRacing.
- Task documentation, observation and action specifications, and prompts for fresh proposals, editing, recombination, and repair.
- A separate-process Docker execution environment for evaluating generated Python policies.
- Archived source code, episode returns, lineage records, model-call logs, and analysis tools.

### 2. Who were the subjects of your study?

- The study involves generated policy programs and simulated environments; there are no human or animal participants.
- Each policy maps observations to actions and may maintain internal state during an episode.
- The main unit of replication is an independently initialized search.
- The planned experiment contains ten searches per environment, for sixty searches overall. Candidates and evaluation episodes are measurements within those searches.

### 3. What was the design of your research?

- A repeated-search experiment measuring performance changes within each search.
- Each search uses ten generations, fifty candidate slots per generation, and an elite table containing up to ten policies.
- Candidates are selected using ten fixed search episode seeds.
- Generation winners are evaluated separately on one hundred held-out episode seeds.
- The primary outcome is the median within-search change in held-out reward from generation one to generation ten, reported separately by environment.
- The design measures the behavior of the complete procedure; it does not isolate the contribution of individual operators or establish superiority over alternative algorithms.

### 4. What procedure did you follow?

1. Supply the language model with the environment description and policy interface.
2. Generate fifty fresh candidates for the first generation.
3. Validate and evaluate candidates, allowing a bounded number of repairs for invalid or failing proposals.
4. Retain the ten highest-scoring valid policies.
5. Generate subsequent populations using fresh proposals, edits of individual elites, and recombinations of multiple elites.
6. Under the standard allocation, use **20% fresh proposals, 40% edits, and 40% remixes**.
7. Repeat until ten generations are complete, preserving checkpoints and failure records.
8. Evaluate each distinct generation winner on the held-out panel without feeding those outcomes back into search.
9. Summarize performance, uncertainty, target attainment, failures, resource use, and representative policy changes.

## Results

### 1. What are your most significant results?

The main experiment is pending in the manuscript. The strongest available **pilot observations** are:

- **LunarLander:** Held-out reward increased by **281.34**, reaching **213.95**, above the target of 200.
- **CarRacing:** Held-out reward increased from **159.04 to 595.37** over ten generations, although it remained below the target of 900.
- **MountainCar:** One completed pilot reached **−102.17**, exceeding the target of −110.
- **CartPole:** The first-generation winner already achieved a held-out reward of **500**.

Present the completed main results through per-environment learning curves and a table of paired generation-one-to-ten changes, with uncertainty and eligible-search counts.

### 2. What are your supporting results?

- Acrobot improved its held-out reward by **35.11**, but its final score remained slightly below target.
- BipedalWalker remained far below its target, showing that successful policy generation was not uniform across environments.
- Held-out performance did not always increase alongside search performance.
- CarRacing and BipedalWalker completed ten search generations but retained failed overall reporting status; some held-out evaluations were missing.
- The pilots used differing stopping points and execution histories, so they cannot substitute for the main-study replicates.

Supporting visuals should include target thresholds, individual search trajectories, and selected policy source and lineage examples. Resource tables should report calls, repairs, discarded candidates, time, and available costs.

## Discussion and Conclusions

### 1. What are the study's major findings?

- The pilot evidence suggests that a small elite table can support the generation and iterative improvement of executable control policies.
- Outcomes vary by task: some policies succeed immediately, some improve through continued search, and others plateau below target.
- The main experiment must establish how consistently these patterns recur across searches.
- The current evidence does not establish that elite reuse outperforms additional independent sampling, or that any particular operator causes the improvements.

### 2. What is the significance/implication of the results?

- A simple evolutionary search procedure is a promising approach to automated policy development and warrants systematic evaluation.
- Executable outputs allow direct inspection of policy behavior and changes across generations.
- Held-out evaluation is essential because improved search scores alone do not demonstrate improved performance on unseen episodes.
- The results concern familiar benchmark environments supplied with task documentation; they do not establish general problem-solving ability.
- This paper provides a foundation for subsequent studies of algorithm comparisons, operator ablations, unfamiliar tasks, and computational efficiency.
