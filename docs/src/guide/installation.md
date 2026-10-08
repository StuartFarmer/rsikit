# Installation and quickstart

The native package targets POSIX systems, including macOS and Linux. `Run` uses
`fcntl` and evaluation workers use POSIX process/signal behavior; importing the
package on native Windows is not supported. Use a Linux environment there.
Python 3.10+ is required. The offline examples and core suite have been checked
on macOS with Python 3.10 and 3.14, including wheel installation outside the
checkout. CI covers both versions on Linux and macOS; other interpreter/dependency
combinations are not independently certified. The Dockerfile defaults to Python 3.14.

## Install from a checkout

```sh
git clone https://github.com/StuartFarmer/rsikit.git
cd rsikit
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m examples.cartpole
```

The last command requires no model credentials. It runs a hand-written controller
for at most 50 steps and prints reward, step count, and episode ending flags.
See [the existing-system walkthrough](../examples/existing-system.md) to continue
into a small search.

## Install a built package

Install a wheel supplied for the revision you intend to use:

```sh
python -m pip install /path/to/rsikit-0.0.1-py3-none-any.whl
rsikit --help
```

The wheel contains `rsikit` and `research`, including the `rsikit` command. The
`examples` package and `scripts/run` are checkout tools; `python -m examples...`
requires a source checkout. A release on a public package index is not assumed
by these instructions.

## Optional dependencies

From a checkout, install only the extras needed by your task:

| Extra | Purpose |
| --- | --- |
| `openrouter` or `openai` | OpenAI-compatible provider SDK used by generation. The unified CLI currently supports OpenRouter. |
| `box2d` | Box2D environments such as LunarLander and BipedalWalker; native build tools/SWIG may be needed. |
| `video` | Rendering and video export dependencies. |
| `ocean` | Python prerequisites for the experimental Ocean integration; it still needs the native setup supplied by the repository. |
| `meta` | Experimental meta-optimization integration dependencies. |
| `dev` | Ruff for development checks. |
| `docs` | Griffe docstring rendering (Python 3.11+); install mdBook 0.5.2 separately for local docs builds. |

For the featured paid search:

```sh
python -m pip install -e '.[openrouter]'
export OPENROUTER_API_KEY='your-key'
```

Use the [existing-system example](../examples/existing-system.md) for an explicit
model and small budget. Generated scores and model latency vary. The
[CLI reference](../api/cli.md) describes model-call and spend limits.

## Docker launcher

From a checkout with Docker running:

```sh
scripts/run examples.cartpole
```

This builds the application image and runs the named Python module. It mounts
`runs/` into `/app/runs`; `RSIKIT_RUNS_DIR` changes the host output directory.
The launcher reads the checkout's `.env` when present, or `RSIKIT_ENV_FILE`, and
forwards supported provider-key environment variables. `RSIKIT_CPUS` and
`RSIKIT_MEMORY` override its defaults of 4 CPUs and 8 GB. Invoking
`research.cli`, Ocean examples, or `unittest` also selects the Ocean image build
and Linux amd64 platform.

The whole application runs in one container. Episode workers are local processes
inside it, not separate Docker containers. Generated Python executes code with the
application's available permissions and credentials. Static source validation,
worker processes, and this container launcher do not constitute a hostile-code
sandbox. Run only code and model outputs you are prepared to execute in that
environment; load saved pickle data only from trusted runs.
