# Manuscript work while the main experiment runs

Working manuscript: [elitetable-policy-search.md](elitetable-policy-search.md).
Frozen reference: [companion draft](../companion/papers/elitetable-policy-search.md).
Edit this working manuscript; leave the frozen launch package intact.

## Central question

How does the held-out quality of search-selected executable policies change
through ten generations, including after search targets are reached?

The eventual claim depends on the complete evidence: improvement, saturation,
regression and failure are all reportable outcomes. A rising retained search
score alone is a consequence of elitism, not evidence of held-out improvement.
This paper describes the combined method; comparative and causal claims require
separate evidence and remain outside the frozen experiment.

## Finish while experiments run

1. Tighten the introduction around this question and the concrete contribution:
   an explicitly specified method, measured trajectories across six tasks, and
   an independently reproducible evidence package. Phrase empirical contributions
   as questions until the full results are available.
2. Expand related work beyond the current FunSearch, AlphaEvolve and Gymnasium
   citations. Verify primary work on programmatic control and language-model
   policy synthesis; explain the actual overlap and differences. Do not claim
   novelty or superiority from the existence of a different implementation.
3. Edit methods for readers unfamiliar with the code. Keep generation barriers,
   operator allocation, selection, ties, repairs and held-out separation in the
   main text; move operational detail and exact prompts to appendices.
4. Finalize experimental-design and limitations prose against the frozen protocol.
   Retain pilot-driven CarRacing selection, conditional seed-panel inference,
   modest search count, provider/hardware dependence, and missing-data rules.
5. Prepare captions and section structure for the evidence below. Leave outcome
   cells empty and avoid choosing a favorable main example during collection.

## Evidence to insert after collection

| Paper element | Source | Completion check |
| --- | --- | --- |
| Algorithm overview and pseudocode | Frozen agent and prompts | Search evaluation and held-out reporting visibly separate |
| Six-task trajectory figure | Main checkpoint winners and raw held-out returns | Individual runs, median/IQR, valid counts and all ten generations |
| Primary endpoint table | Paired generation-10 minus generation-1 returns | Whole-search bootstrap, eligible denominators, no missing-as-zero |
| Target and failure reporting | Inventory, checkpoints, attempts and evaluation failures | Any-generation search vs generation-10 held-out target distinguished |
| CarRacing case study | Prespecified representative, generations 1/5/10 and ancestry | Median-based selection; all missing source/checkpoints disclosed |
| Resource table | Call logs, candidate histories and attempt elapsed times | Repairs and unknown usage visible, no double-counting resumes |

Then write results and discussion, replace the provisional abstract and conclusion,
and independently check every numerical statement against its artifact. Repeat
clean offline reproduction using the main archive before publication.

## Author decisions

- Target audience/venue and page limit; until chosen, keep a venue-neutral manuscript.
- Author names, order, affiliations and acknowledgements.
- Public repository location, project license and third-party notices before release.

These decisions do not block methods writing or the running experiment.
