# EliteTable — Level 1 Outline

Based on the [working manuscript](elitetable-policy-search.md), 25 September 2026.

## 1. What is the topic of my paper?

EliteTable: a simple evolutionary search method that uses a language model to generate executable policies, retain a small table of strong solutions, and improve them through edits, recombination, and fresh proposals.

## 2. Why is this topic important?

Automating the design of problem-solving algorithms could reduce the human effort needed to develop effective solutions. This paper investigates how much a relatively simple search procedure can accomplish while producing source code that people can inspect and run without further language-model calls.

## 3. How could I formulate my hypothesis?

**Hypothesis:** Iteratively generating, evaluating, and reusing elite policy programs improves their performance on unseen episodes within a ten-generation search budget.

The specific test is whether generation-ten winners achieve higher held-out rewards than generation-one winners. For each environment, the primary measure is the median paired change across eligible searches.

## 4. What are my results (include visuals)?

The main-study results are still pending in the draft. The available **exploratory pilots** show:

| Environment | Observed pilot result |
| --- | --- |
| CartPole | Reached a held-out reward of **500** in generation one. |
| MountainCar | One completed pilot reached **−102.17**, exceeding the **−110** target. |
| Acrobot | Held-out reward improved by **35.11**, but finished slightly below its target. |
| LunarLander | Held-out reward improved by **281.34**, reaching **213.95** and exceeding its target. |
| CarRacing | Held-out reward increased from **159.04 to 595.37**, remaining below the **900** target. |
| BipedalWalker | Final held-out reward was **−8.17**, far below the **300** target. |

These pilots have different stopping points and execution histories. CarRacing and BipedalWalker completed ten search generations but have failed overall reporting status; CarRacing also has missing intermediate evaluations. CartPole's execution timeout changed on resume. These runs are preliminary evidence, separate from the planned main experiment.

Sources: [pilot inventory](../figures/target-and-progress.csv), [completed-pilot summaries](../figures/results.csv), and [preparation audit](../outputs/paper1-audit-2026-09-25.md).

![Exploratory pilot results: search reward, held-out reward, and task targets across generations.](../figures/target-and-progress.png)

*Figure 1. Exploratory pilot trajectories. Blue: search reward. Orange: held-out reward. Green: task target. This figure includes additional exploratory environments outside the six-task main suite. Missing held-out measurements remain gaps; failed run status does not mean zero reward.*

## 5. What is my major finding?

**Preliminary finding:** The pilot evidence suggests that a simple elite-table search can produce effective executable policies and improve held-out performance on several control tasks, although success varies substantially by environment.

The abstract's stronger statement that EliteTable “solves most” environments within ten generations still needs the main-study results.
