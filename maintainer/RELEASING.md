# Releasing rsikit

The repository currently prepares **0.0.1**, an experimental release candidate.
Do not equate a successful documentation build with permission to publish every
file or with verified search quality. Track evidence in
[RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md) and outstanding work in the
[checklist](../RELEASE_CHECKLIST.md).

## Before publication

- Publish the clean source snapshot selected by the maintainer, not the original
  development history. It excludes `data/forex/` and all Git history; the original
  private checkout retains them. Do not push that history to the public repository.
- Keep Bitcoin's CC BY-NC 4.0 attribution distinct from the Apache-2.0 license on
  original code. If the intended distribution requires only permissive data,
  replace/remove that dataset in a separately reviewed change.
- Review historical local-path disclosures noted in the verification record. A
  new public snapshot is an alternative to publishing private development history.
- Run one deliberately budgeted live-provider smoke test if the release advertises
  the paid workflow as verified. Offline tests check protocol behavior, not provider
  availability or paper results. Supply credentials locally, never in artifacts.
- Review the rendered site in a browser, including narrow screens, keyboard
  navigation, search, copyable snippets, and API signatures.
- Confirm package-index name/version availability and maintainer publishing access.
  Configure GitHub Pages to use Actions and enable private vulnerability reporting.

## Build the candidate

The prepared `rsikit-public-0.0.1.tar.gz` archive contains a clean source tree
without Git history. Verify it against the accompanying `SHA256SUMS`, extract
it into a new directory, and initialize a new repository there when ready.
Do not copy the original checkout's `.git` directory into it.

From the reviewed revision in a fresh checkout:

```sh
uv sync --locked --extra dev --extra docs
uv run --no-sync ruff check rsikit research examples tests scripts
uv run --no-sync ruff format --check rsikit research examples tests scripts
uv run --no-sync python -m unittest tests.test_release_examples tests.test_package_boundaries -v
cargo install mdbook --version 0.5.2 --locked  # once
mdbook build
uv run --no-sync python scripts/check_docs.py
uv build --out-dir dist/release
uv run --no-sync python scripts/check_distribution.py dist/release
uvx twine check dist/release/*
```

Use an empty `dist/release` directory. Existing local `dist/` files can belong to
older versions; never upload a wildcard that includes unrelated artifacts.
Run the complete core CI matrix and the manually triggered native integration
workflow before tagging. To reproduce the latter locally:

```sh
docker build --build-arg OCEAN=1 --platform linux/amd64 -t rsikit:release-check .
docker run --rm --platform linux/amd64 --network none --user 1000:1000 -e HOME=/tmp rsikit:release-check \
  python -m unittest discover -s tests -v
```

The optional Maze tests need `scripts/install_maze.py` before an offline test run.
The pinned source/license installer is separate from the Ocean build. Native
optional checks are not part of the fast contributor suite.

Install the candidate wheel in a clean environment **outside the checkout** and
run `rsikit --help` plus copies of both featured scripts. Verify installed prompt
resources and that results can be reopened. Repeat on supported platforms; record
the interpreter and dependency versions rather than asserting every combination.

## Publish the reviewed revision

1. Complete the checklist and merge reviewed changes. Set the version in
   `pyproject.toml`, the documentation landing page, and changelog together.
2. Tag that exact revision. Build the final wheel and sdist from it and retain
   checksums plus validation results. Upload only those reviewed files using the
   maintainer's configured package-index credentials/trusted publishing setup.
3. Push that reviewed revision to `public-release`. **Publish documentation** runs
   automatically on each push to that branch. It builds only the curated
   public tree, checks rendered API coverage, and publishes via the `github-pages`
   environment. Configure an environment review gate if desired. The source/edit
   links use the selected commit SHA.
4. Verify installation from the public index and the live docs at
   `https://stuartfarmer.github.io/rsikit/`. Set that URL in the repository About
   section and link the release notes. Verify old published URLs separately if a
   previous public site existed; the retained GitHub forwarding pages are not HTTP
   redirects for an older website.

No workflow here automatically uploads to PyPI or makes the repository public.
Push to `public-release` only when the documentation is ready to publish.
