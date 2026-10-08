# rsikit

Generate, evaluate, and optimize Python policies in Gymnasium environments.
rsikit provides policy definitions, episode evaluation, process workers, and
saved runs. Included research optimizers build on the same small interface.

| Documentation | Purpose |
| --- | --- |
| [Guide](docs/src/guide/introduction.md) | What rsikit is, its concepts, and how to use it. |
| [API reference](docs/src/api/index.md) | Classes, functions, arguments, and lifecycle contracts. |
| [Examples](docs/src/examples/index.md) | Try an existing system and implement a custom one. |

## Quickstart

From a clone, on Linux or macOS, with Python 3.10+ and
[uv](https://docs.astral.sh/uv/):

```sh
uv venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m examples.existing_system
```

This runs a small EliteSearch workflow on CartPole using a deterministic offline
provider. It makes no API calls and saves the winner and its held-out evaluation
under `runs/`. See [the walkthrough](docs/src/examples/existing-system.md) to use an
OpenRouter model, or [build a custom optimizer](docs/src/examples/custom-system.md).

The wheel includes `rsikit`, the `research` algorithms, and the `rsikit` CLI.
Examples and the Docker launcher require a clone or source distribution.
See [installation](docs/src/guide/installation.md) for extras and Docker support.

Version 0.0.1 is experimental; APIs may change. Native imports and workers require
POSIX (Linux/macOS). Generated Python executes with the application's permissions;
see [execution boundaries](docs/src/guide/policies.md). Research implementations are
adaptations, not claims of reproduced paper results.

## Development

```sh
uv pip install --python .venv/bin/python -e '.[dev,docs]'
.venv/bin/python -m unittest tests.test_release_examples -v
cargo install mdbook --version 0.5.2 --locked  # once, if mdbook is not installed
mdbook serve --open
```

Docs use mdBook 0.5.2 and Python 3.11+ for docstring rendering. Preview at
`http://localhost:3000/`; `mdbook build` writes the site to `book/`.

[Contributing](CONTRIBUTING.md) covers tests and documentation authoring.
[Release notes](CHANGELOG.md), [release checklist](RELEASE_CHECKLIST.md), and
[release procedure](maintainer/RELEASING.md) track publication readiness.
Report bugs through [GitHub issues](https://github.com/StuartFarmer/rsikit/issues);
see [SECURITY.md](SECURITY.md) for sensitive reports.

## License

Original code and documentation: [Apache-2.0](LICENSE). Third-party code, fonts,
images, and datasets retain their own terms in [NOTICE](NOTICE). In particular,
the bundled Bitcoin data is CC BY-NC 4.0, not Apache-2.0. The forex export is
excluded from package builds and the public source snapshot.
