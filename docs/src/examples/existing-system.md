# Try an existing system

This example runs the included **EliteSearch** optimizer on CartPole, saves a
winner, reloads its source, and measures it on two held-out seeds. The default
provider returns two fixed candidate programs, so the whole journey works offline.

## Prerequisites and commands

Use a macOS or Linux checkout with the [base installation](../guide/installation.md).
The optional model-install command below uses `uv`, as in the repository README;
if you created the environment with `python -m venv`, you can instead use
`.venv/bin/python -m pip install -e '.[openrouter]'`.
First check the environment without a worker or API key:

```bash
.venv/bin/python -m examples.cartpole
```

It prints `reward=50.0 steps=50 terminated=False truncated=True` with the current
CartPole implementation. Then run the optimizer from the repository root:

```bash
.venv/bin/python -m examples.existing_system --output runs/existing-demo
```

Choose a new output path for each invocation; `Run.create` will reject an existing
directory. Omit `--output` to create a unique timestamped directory automatically.
The offline example usually takes a few seconds and costs nothing. A verified
Python 3.14/macOS run finished in under a second and ended with:

```text
training_mean=100.0
heldout_mean=100.0
winner=.../runs/existing-demo/best.py
```

Additional progress logs show each candidate's status and scores. Success means
the command exits cleanly, writes `best.py`, and prints both measured means.
Scores from a live model are not guaranteed to match the offline result.

## Budget and model setup

The budget is one generation of two candidates, evaluated on seeds `0` and `1`.
Each episode is capped at 100 steps. Selection keeps one elite, which is then
evaluated on seeds `100` and `101`. That is at most six episodes, one episode
worker, and no repair attempts. This tiny budget demonstrates initial selection;
increase generations for EliteSearch's subsequent edit/remix rounds.

To replace the fixed responses with OpenRouter, install its extra and supply a
model ID supported by your account:

```bash
uv pip install --python .venv/bin/python -e '.[openrouter]'
export OPENROUTER_API_KEY='your-key'
.venv/bin/python -m examples.existing_system --model 'provider/model-id' --output runs/existing-live
```

`provider/model-id` is a placeholder, not a working model name. This makes up to
two model requests, each capped at 2,048 output tokens and 60 seconds. Input tokens
also incur charges; cost depends on the model and provider pricing. Allow seconds
to a few minutes. The script sets Slick's template root to the installed
`research.elitesearch/prompts` directory. No extra prompt files are needed.

Generated Python executes in local worker processes. Use a provider and generated
code you trust: workers and static validation do not provide a hostile-code sandbox.
The paid variant has not been smoke-tested for this release; the automated check
uses only the fixed offline responses.

## Script

Edit `examples/existing_system.py`; this page embeds that same file.

```python
{{#include ../../../examples/existing_system.py}}
```

## Read the result

`run.sqlite` contains policies, scores, and EliteSearch records; `exports/` contains
all evaluated candidate sources; `episodes/` contains the raw training and held-out
trajectories. `best.py` is the selected policy with identity metadata. The script
loads it with `PolicyDefinition.from_file` before held-out evaluation, so this
also checks the saved policy can run.

The reported mean is the mean fitness over each seed panel. CartPole has no
fitness override, so this equals mean cumulative reward. A failed candidate's
partial reward is excluded from selection. Candidate errors are printed, and a
run with no successful candidate raises an error. Held-out errors also fail the
command rather than presenting a partial mean as success. Infrastructure failures
propagate immediately.

The context managers close the original environment, Run, and Executor; each
worker owns its policy and environment copy. See [runs and recovery](../guide/runs.md)
for inspecting saved data and [optimization](../guide/optimization.md) for larger
budgets. Only load episode pickle files from trusted runs.
