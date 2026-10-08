# Python API

The core API is available directly from `rsikit`. Choose the guide for a first
working example; use these pages for signatures, lifecycle rules, and failures.

| Import | Reference | Guide |
| --- | --- | --- |
| `Policy`, `PolicyDefinition`, `PolicyEncoder`, `generate` | [Policies and generation](policies.md) | [Writing policies](../guide/policies.md) |
| `Episode`, `Evaluator`, `evaluate` | [Evaluation](evaluation.md) | [Introduction](../guide/introduction.md) |
| `Job`, `Executor`, `execute` | [Execution](execution.md) | [Installation](../guide/installation.md) |
| `Optimizer`, `search` | [Optimization contract](optimization.md) | [Optimization](../guide/optimization.md) |
| `Run` | [Run storage](runs.md) | [Runs](../guide/runs.md) |

Concrete optimizers and experiment configuration live in the `research` package:
see [optimizers](optimizers.md) and the [CLI reference](cli.md).
For integration examples, see [an existing system](../examples/existing-system.md)
and [a custom system](../examples/custom-system.md).

## Ownership and failure boundaries

`PolicyDefinition` stores source without executing it. `evaluate` uses existing
policy and environment objects, leaving their cleanup to the caller. `execute`
and `Executor` serialize jobs into worker processes and clean up worker copies.
The original inputs remain caller-owned.

Policy errors and worker timeouts normally return an `Episode` with `error` set.
Infrastructure failures raise exceptions. `Run.mean_scores` also raises when any
requested episode fails, so a failed panel cannot silently become a mean score.
See [evaluation exceptions](evaluation.md#exceptions).

Workers execute Python with the host process's privileges. Process isolation is
not a security sandbox. Persistent queues and run directories must be trusted:
they contain serialized Python objects.
