# Examples

Run these from the repository root after the [clone installation](../guide/installation.md).
The scripts require a POSIX system (macOS or Linux) for native worker execution.
No rendering, Docker, or native simulator extras are needed.

| Journey | Outcome | Command from the checkout | Time and cost |
| --- | --- | --- | --- |
| Installation check | Evaluate a handwritten CartPole policy | `.venv/bin/python -m examples.cartpole` | Usually under a second; no key or charge |
| [Try an existing system](existing-system.md) | Run EliteSearch, save its winner, and test unseen seeds | `.venv/bin/python -m examples.existing_system` | Usually a few seconds offline; no key or charge |
| [Implement a custom system](custom-system.md) | Write a tiny optimizer and compose the shared evaluation loop | `.venv/bin/python -m examples.custom_system` | Usually a few seconds; no key or charge |

The installation check prints reward, episode length, and termination flags. Both
optimization journeys print `training_mean`, `heldout_mean`, and a path to `best.py`.
Their output directories default to new timestamped folders under `runs/`.

The existing-system page also shows an explicit OpenRouter option. That mode needs
an API key and incurs the selected provider's charges. Offline runs and the
automated smoke checks make no network model calls.

The other scripts in `examples/` are research utilities; start with these two
journeys before using larger experiment budgets or optional environments.
