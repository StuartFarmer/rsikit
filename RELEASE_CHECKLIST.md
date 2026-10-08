# rsikit public release checklist

Release goal: a Python developer can understand rsikit, run an existing system,
and implement a custom one using three documentation sections: **Guide**,
**API reference**, and **Examples**. Match Keras's organization at rsikit's scale.

Implementation is prepared on `docs/public-release-checklist`. See
[verification results](maintainer/RELEASE_VERIFICATION.md) and the
[release procedure](maintainer/RELEASING.md). Unchecked items still require work;
this is not a claim that publication has occurred. The initial review
used local `main` at `f0f0fc9` (2026-10-08), including four commits ahead of
`origin/main`. Existing documentation is unverified source material: it mixes
user documentation, historical behavior, implementation plans, and experiment notes.

## 1. Establish what is current

- [x] Inventory the README, every tracked documentation page, and example. Record
  its purpose, whether its claims match code/tests, and its disposition: retain,
  rewrite, merge, archive, or remove. Inspect ignored local files separately;
  their presence in a checkout does not make them part of the release.
- [x] Verify retained public commands, signatures, defaults, imports, installation steps, and
  lifecycle claims against the release code. Resolve contradictions before moving
  prose into the new site; a recently edited page is not necessarily accurate.
- [x] Move development plans and design scribbles, including `docs/superpowers/`,
  outside the published documentation tree. Keep useful history in a clearly
  labeled maintainer archive; do not publish it as product guidance.
- [x] Separate experiment reports, benchmark JSON, legacy Ocean notes, and research
  proposals from user documentation. Keep selected research reports accessible
  through contextual links, with dates and the code/configuration they describe.
- [x] Remove obsolete instructions and duplicate explanations after their useful
  content has been verified and migrated. Repair incoming links or retain small
  forwarding pages for established URLs.

Done when each existing document has a deliberate destination and no unverified
page becomes public documentation merely because it already exists.

## 2. Guide — what rsikit is and how to use it

- [x] Write a short introduction defining rsikit, its intended users, and its
  current scope: generating and evaluating Python policies, composing optimization
  loops, and saving runs. Distinguish the shared library from included research
  implementations; explain what “self-improvement” means in those implementations.
- [x] Establish one vocabulary: environment/task, runtime `Policy`, source
  `PolicyDefinition`, episode, evaluation, optimizer, proposal round, and run.
  Explain how these fit together before introducing individual algorithms.
- [x] Explain the main flow: propose definitions → evaluate on a seed panel →
  return episodes → update optimizer → select a best policy. Distinguish rewards,
  fitness, and aggregate scores; document the final-info `fitness` override.
- [x] Provide installation and first-run instructions with tested Python/OS
  support. Separate installed-package usage, clone-only examples, optional extras,
  and the Docker launcher. State the native executor's POSIX requirement.
- [x] Explain evaluating an existing policy, generating a policy, running a
  search, and inspecting results. Put complete runnable walkthroughs in Examples
  and link to them rather than repeating large programs throughout the Guide.
- [x] Explain ownership and lifecycle: direct evaluation uses caller-owned
  instances; worker execution uses private copies; `Executor` requires an async
  context; callers own cleanup. Distinguish `evaluate`, `execute`, and
  `Run.evaluate` by their jobs, inputs, and outputs.
- [x] Explain seeds, proposal/episode budgets, generation versus evaluation
  concurrency, policy failures versus infrastructure failures, and cancellation.
  Explain training versus held-out evaluation without promising deterministic LLM
  output or comparable work per “generation” across algorithms.
- [x] Explain saving, loading, episode reuse, and resuming. Document that reuse
  assumes a fixed environment/configuration within a Run, and distinguish reopening
  a run or queue from restoring an optimizer checkpoint. State resume support per
  algorithm/variant rather than implying universal support.
- [x] Explain extension points: custom `Policy`, Gymnasium environment, and
  `Optimizer`. Distinguish the Python optimizer protocol from the CLI adapter
  contract (`Options`, `add_arguments`, `optimize`).
- [x] Document actual execution boundaries near the relevant setup steps:
  source validation is a static contract check; generated Python executes code;
  worker processes and the application container do not imply a hostile-code
  sandbox; saved pickle data must be trusted. Replace stale Docker-worker wording
  and reconcile it with the current application-container model.

Suggested Guide pages: **Introduction and concepts**, **Installation and quickstart**,
**Policies and evaluation**, **Optimization**, **Runs and recovery**, and
**Customization**. Reuse verified material from `INNER_LOOP.md`, `RUNS.md`,
`CLI.md`, and `IN_PROCESS_SANDBOX.md`.

## 3. API reference — precise contracts from docstrings

- [x] Define the supported public API before generating pages. Start with
  `rsikit.__all__`; explicitly decide which environment helpers, exceptions,
  progress hooks, and `research.*` classes/configuration objects are public.
  Keep private workers, database models, and implementation helpers out of the
  reference unless they are deliberate extension points.
- [x] Make docstrings the source of truth for API contracts. Fill in purpose,
  import path, signature, parameter types/defaults, return values, exceptions,
  side effects, async/context requirements, and a short usage example where useful.
  Current core docstrings are often only summaries.
- [x] Cover policies: `Policy` and its lifecycle methods, `PolicyDefinition`
  fields/identity/load/save/validation, and `PolicyEncoder`.
- [x] Cover evaluation: `Episode` fields and helpers, `Evaluator`, `evaluate`,
  `Job`, `Executor.execute`/`iterate`, and standalone `execute`. Specify result
  shapes, partial/failure behavior, timeouts, copying, and serialization limits.
- [x] Cover optimization and storage: `Optimizer.done`/`best`/`propose`/`update`,
  `search`, `generate`, and public `Run` methods, including `collect`, `evaluate`,
  `mean_scores`, persistence, and loading. Specify complete-round feedback as
  `{policy_id: {seed: Episode}}` and distinguish episode feedback from mean scores.
- [x] Give supported research optimizers and configuration objects concise
  reference pages using their real import paths. Label experimental variants and
  identify which are exposed by the unified CLI versus Python/example entry points.
- [x] Keep CLI flags and YAML configuration in a dedicated reference page linked
  from the Guide. Check it against parser/config models and `--help`; Python
  docstring generation alone will not document this surface adequately.
- [x] Generate API pages with an established docstring-capable documentation tool.
  Use a curated navigation list, source links, stable anchors, and links to relevant
  guides/examples. Do not maintain handwritten copies of signatures/defaults or
  build a custom Keras-style documentation generator.

Done when a reader can look up every supported public object without searching
an experiment report or reading its implementation.

## 4. Examples — two complete user journeys

- [x] Create an Examples index with a one-line outcome, prerequisites, exact run
  command, expected output, approximate runtime, and API-key/cost requirements
  for each featured example. Keep larger benchmark utilities out of the beginner path.
- [x] **Try an existing system:** offer the existing `examples/cartpole.py` as a
  no-key installation check, then walk through one included optimizer on CartPole
  with a small explicit budget. Choose the optimizer after verifying its shortest
  supported entry point; the current README opens with an Ocean experiment that
  requires additional native setup.
- [x] In that walkthrough, show provider/model setup, the task and seed panel,
  execution, where outputs are written, how to read the score and errors, and how
  to load and evaluate the saved winner on held-out seeds. Show the observable
  success conditions without promising a particular generated score.
- [x] **Implement a custom system:** provide a minimal runnable optimizer exposing
  `done`, `best`, `propose()`, and `update(results)`, composed with `search`,
  `Run.evaluate`, and `Executor`. Use a tiny deterministic candidate source so
  learning the contract does not require paid calls; explain where an LLM proposal
  strategy can replace it.
- [x] Include a small custom `Policy` in that example and explain state reset,
  action validity, and saving a definition. Link to a custom-environment recipe;
  a policy subclass by itself does not demonstrate a custom optimization system.
- [x] Give both journeys complete scripts with imports and `asyncio.run(...)`.
  Keep scripts as the editable source and embed or link them from documentation;
  avoid independently maintained script, Markdown, and notebook copies.
- [x] Add a small automated smoke check for each featured journey using offline
  inputs/providers; ordinary docs builds and CI must not make model calls.
- [ ] Verify the paid variant manually before release and record the version/settings.

Done when a fresh user can complete both journeys without reading implementation
code, supplying undocumented files, or inferring missing async setup.

## 5. Repository and distribution cleanup

- [x] Replace blanket `.gitignore` entries for `docs`, `research/`, and `data/`
  with rules for actual generated/private outputs. These directories contain
  tracked release inputs, while new files are currently hidden by default.
- [x] Choose and add a top-level license and package license metadata. Review
  included algorithm notices, fonts, card images, and data provenance/redistribution
  terms; retain their required attribution in source and built distributions.
- [x] Reduce the README to purpose, one working quickstart, the three documentation
  links, supported installation, and project status. Move verified detailed usage
  into the Guide and remove the local sibling `elitelist_papers` link from public
  onboarding. Reconcile contradictory wheel/examples claims.
- [x] Define release scope and stability explicitly: shared core, packaged research
  algorithms, optional integrations, and experimental components. Preserve the
  existing core/research dependency boundary; a public release need not rename or
  reorganize working Python packages.
- [x] Build and inspect both wheel and source distribution. Confirm the CLI target
  `research.cli:main`, prompt templates, runtime text files, environment assets/data,
  and notices are present. Check Hatch's handling of ignored directories rather
  than assuming configured package/include lists guarantee inclusion.
- [x] Install the wheel into a clean environment outside the checkout and verify
  public imports, `rsikit --help`, a no-key evaluation, and template/resource loading.
  Separately verify documented clone/example commands and optional installation
  paths. Resolve the Python `>=3.10` claim against dependencies and actual tests;
  the Docker image currently defaults to Python 3.14.
- [x] Add concise contributor setup/check commands, issue and security-reporting
  routes, release notes, and a release procedure. Add the documentation URL
  to package metadata.
- [ ] Add the documentation URL to the public repository description.
- [x] Review tracked content and history for credentials, private endpoints,
  machine-specific paths, and unintended run artifacts before making the repo
  public. Ignoring `.env` only prevents some future additions.

## 6. Publishable docs and release checks

- [x] Add one site configuration with **Guide / API reference / Examples** as its
  main navigation, a useful landing page, search, and readable code blocks. Build
  from `docs/src/` and its explicit `SUMMARY.md`, keeping plans/archives outside
  the book source directory and search.
- [x] Provide one documented local preview command and one reproducible docs build.
  Keep documentation dependencies separate from runtime dependencies; builds must
  work without credentials, private files, or native optional integrations.
- [x] Add CI for the existing lint/tests, package installation smoke check, featured
  offline examples, and docs build/internal links. Use supported Python versions;
  keep expensive Docker/Ocean checks separate from the basic contributor path.
- [x] Check the rendered site for broken internal links, unresolved API objects,
  constructor signatures, stale imports, and excluded archival pages.
- [ ] Review the rendered site visually, including mobile navigation and keyboard access.
- [ ] Perform a fresh-install walkthrough of both featured examples against the
  release artifact and rendered docs. Fix every undocumented prerequisite found.
- [ ] Publish package and docs from the same release revision, with a visible
  version and matching source links. Verify the live docs and public installation
  command after publishing; retain useful old documentation URLs where practical.

Suggested execution order: inventory and release scope → verify contracts → write
the two examples → consolidate Guide and docstrings → build the site → validate
distributions and publish. A large example gallery, custom theme, separate docs
repository, and notebook conversion pipeline can wait for demonstrated demand.

## Keras format references

- [Model API page](https://keras.io/api/models/model/): signatures, contracts,
  source links, and related guide/example links.
- [API generator](https://github.com/keras-team/keras-io/blob/master/scripts/docstrings.py):
  confirms that Keras renders inspected signatures and docstrings.
- [Guides](https://github.com/keras-team/keras-io/tree/master/guides) and
  [examples](https://github.com/keras-team/keras-io/tree/master/examples): separate
  concept/workflow teaching from concrete applications.
- [Authoring workflow](https://github.com/keras-team/keras-io/blob/master/README.md):
  executable Python is the source for rendered tutorials. Adopt the single-source
  principle; rsikit does not need Keras's full publishing pipeline.

## Remaining publication decisions

- Apache-2.0 has been added; third-party notices are retained. The license item
  is complete for the selected public snapshot: the forex export and Git history
  are excluded, and the original private checkout is preserved. Bitcoin retains its CC BY-NC 4.0 terms.
- Both featured journeys have automated offline checks. A live paid-provider
  smoke test has not been run; no credentials or model budget were used.
- mdBook build, API rendering checks, and package installation checks are
  automated. Visual/mobile/keyboard review and the full human walkthrough remain.
- Repository visibility, GitHub Pages settings, release tagging, package upload,
  live-site checks, and repository About metadata have not been changed.
