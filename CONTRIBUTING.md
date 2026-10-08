# Contributing

Use an up-to-date `main` for new task branches and preserve local-only commits.
Keep shared code in `rsikit/` independent of `research/` and `examples/`.
Algorithms may use the core, but must not import another algorithm.

## Setup and checks

```sh
uv venv
uv pip install --python .venv/bin/python -e '.[dev,docs]'
.venv/bin/python -m unittest tests.test_release_examples tests.test_package_boundaries -v
.venv/bin/ruff check rsikit research examples tests scripts
.venv/bin/ruff format --check rsikit research examples tests scripts
cargo install mdbook --version 0.5.2 --locked  # once
mdbook build
.venv/bin/python scripts/check_docs.py
```

The CI core suite needs no API keys. The optional integration suite includes
native Ocean, video, Box2D, and scientific-library checks; run it with the existing
application launcher:

```sh
./scripts/run unittest discover -s tests -v
```

The launcher builds the image and installs upstream native dependencies. It
requires Docker and network access. Do not add paid model calls to ordinary tests
or documentation builds. Use scripted providers for tests; record live experiment
settings separately when measuring search quality.

## Documentation

The published documentation contains three sections:

- `docs/src/guide/`: concepts and workflows, verified against current code.
- `docs/src/api/`: curated API comments rendered with Griffe and CLI/config reference. Edit
  Google-style docstrings on the Python object to change its API contract.
- `docs/src/examples/`: walkthroughs embedding the runnable scripts in `examples/`
  through mdBook’s native `{{#include ...}}` directive. Edit the script rather than copying its code.

Run `mdbook serve --open` to preview at `http://localhost:3000/`. The API
preprocessor uses `uv` and the locked docs extra (Python 3.11+). `book.toml`
reads only `docs/src/`; `SUMMARY.md` defines the chapter order. Keep drafts and
archives outside that source directory: mdBook also copies non-Markdown assets.
`mdbook build` resolves APIs and embeds examples; `scripts/check_docs.py` checks
the generated assets, internal links, anchors, API coverage, and search scope.
Python/docstring and example edits trigger preview rebuilds.

To include a public object in an API page, use a comment such as:

```markdown
<!-- api: rsikit.policy.Policy
{members: [reset, act, close], inherited_members: false}
-->
```

The comment selects the object and members; signatures and descriptions still
come from Python. `scripts/mdbook_api.py` connects Griffe’s Markdown renderer to
mdBook and resolves cross-references to the documented objects.
Document one concept once and link to it elsewhere. Every public API change
should update its docstring and relevant example/guide in the same change.

Plans, design notes, and dated experiment reports belong under `maintainer/`,
outside the public site. Historical archives and private local drafts are retained in the original
development checkout and are not included in this public snapshot.

Include a focused regression check for behavior changes. Reuse unittest and
existing helpers. Explain the user-visible change and checks run in your PR.
