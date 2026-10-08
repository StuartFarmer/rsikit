# Release cleanup verification

Candidate: rsikit 0.0.1. Worktree based on `f0f0fc9`; checks performed 2026-10-08.
This record distinguishes preparation from publication. No package upload,
repository-visibility change, live model request, or docs deployment was performed.

## Verified

| Check | Result |
| --- | --- |
| Historical documentation preservation | All 63 previously tracked docs files preserved byte-for-byte in the private checkout's maintainer archive. |
| Core Python changes | AST comparison confirms all 12 modified core/environment modules differ only in docstrings. |
| Ruff | Source, research, examples, tests, and scripts pass lint and formatting. |
| mdBook build | mdBook 0.5.2 builds all 19 content pages; rendered checks cover internal links, assets, API object resolution, and both included scripts. |
| Rendered documentation check | Constructors and inherited AlphaEvolve protocol members render; search contains only the curated public sections. |
| mdBook preview | Homepage, Guide, API, Examples, and 16 referenced local assets return HTTP 200 at the local preview; the clean public snapshot also builds and passes all rendered checks. |
| Core CI selection | 115 tests pass on macOS Python 3.10.18, 3.14.2, and 3.14.7. |
| Full native integration suite | Linux amd64 application image, non-root user, network disabled: 455 tests run, OK with 17 optional skips. |
| Clean wheel installation | Both offline example scripts and the CLI verified outside the checkout on Python 3.10.18; both scripts verified on 3.14.7. |
| Package contents | Wheel and sdist pass `scripts/check_distribution.py`; includes runtime prompts, text/data/font resources, and licenses; excludes forex, local drafts, and maintainer archive. |
| Package metadata | Wheel and sdist pass `twine check`; `uv lock --check` passes. |
| Clean public snapshot | Rebuilds wheel/sdist with content and metadata checks passing; featured-example and package-boundary tests pass (3 tests); `uv sync --locked --extra dev --extra docs` and docs build/rendered checks pass. |
| Maze installer | Pinned source compiled on macOS; installed LICENSE/NOTICE match retained originals; all three native Maze tests pass on Python 3.10.18. |
| Workflow configuration | Three workflows parse; read-only default permissions; docs publish on pushes to `public-release`, with deployment permissions granted only to the deploy job. |

Optional checks report explicit skips when their runtime facilities are unavailable;
passing tests do not establish search-quality or paper-reproduction claims.

## Review fixes

- Migrated to mdBook 0.5.2 at the maintainer’s request. Guide/API/Examples live
  under `docs/src/`, separate from private drafts. API comments use Griffe’s
  Markdown renderer; examples use native includes. CI, packaging, preview
  instructions, and release artifacts use mdBook.

- Fixed the public-tree exclusion rule also excluding MkDocs theme assets.
  The rendered check now requires styles, scripts, fonts, and search assets;
  local preview requests for CSS, JavaScript, search, and favicon return HTTP 200.
- Corrected a test's assumption that two short jobs must use two distinct workers.
  Queue scheduling may reuse one worker; private-copy and parent-PID checks remain.
- Excluded the maintainer archive explicitly from Hatch's sdist. Unanchored
  include patterns had also selected nested historical `docs/` and README files.
- Enabled merged constructor signatures and selectively enabled inherited optimizer
  members. Added `check_docs.py` to catch these omissions in rendered output.
- Retained PufferLib's identical MIT license at both pinned Ocean revisions and
  added a Maze installer step to copy its license and adaptation notice.

An independent reviewer found the archive and API-rendering issues; they were
fixed and verified. The small uv/pip prerequisite mismatch in the example page
was also clarified. No runtime redesign was needed.

## Environment findings

The original local full-suite attempt lacked the `control` dependency and hit
macOS video-worker errors. The pre-existing Docker image lacked Huey, so a fresh
image was built. Mounting host bytecode into that image caused marshal errors;
the final mounted-source check uses a separate bytecode-cache path. Running the
container as root violates existing test fixtures' non-root assertion; the
integration workflow and documented command now use a non-root account.

A clean managed CPython 3.14.0 environment stalled in worker startup and was
stopped. The same resolved dependency set works with Homebrew CPython 3.14.7.
Use the tested current patch releases; the old managed build is not certified by
these checks. This observation is not an attribution of the fault to a particular
upstream component.

## Publication scope

The maintainer selected **exclude forex and prepare a clean public snapshot**.
That snapshot omits the forex export, Git history, personal AGENTS instructions,
archived plans/benchmarks, and local drafts. The original checkout/history is
preserved. Publish from the snapshot rather than changing visibility on the
existing development repository.

Pattern scans over current content and 63 reachable commits found no recognizable
provider keys, private-key blocks, or password-bearing URLs. The generic secret
match was a deliberate test fixture. This is not a guarantee against every secret
format. Historical workstation paths and old notebooks remain private with the
original history rather than being included in the snapshot.

Original code/docs use Apache-2.0. Third-party licenses remain intact. Bitcoin's
Coin Metrics data retains CC BY-NC 4.0 and is explicitly distinguished in NOTICE.

Still pending: visual/mobile/keyboard review, a deliberately budgeted live-provider
smoke test, public index/account setup, and actual release/docs publication.
