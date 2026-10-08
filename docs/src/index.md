# rsikit

Generate, evaluate, and optimize Python policies in Gymnasium environments.
rsikit supplies the policy lifecycle, episode evaluation, process workers, and
saved runs. Included research optimizers use those pieces to search over policy
source code.

| Start here | What you will find |
| --- | --- |
| [Guide](guide/introduction.md) | Concepts, setup, evaluation, optimization, and customization. |
| [API reference](api/index.md) | Signatures, arguments, results, and lifecycle contracts generated from source docstrings. |
| [Examples](examples/index.md) | Run an existing system and implement a custom one. Both have an offline path. |

New to the project? Follow [installation](guide/installation.md), then
[try an existing system](examples/existing-system.md). To use your own search
strategy, follow [build a custom system](examples/custom-system.md).

Version **0.0.1**, experimental: APIs may change. Native execution targets
Linux and macOS; use the application container on other hosts. Research
implementations are local adaptations, not claims of reproduced paper results.

The project is licensed under Apache-2.0. Bundled third-party assets and data
retain their own terms; notably, Bitcoin price data is CC BY-NC 4.0. See the
[license and notices](https://github.com/StuartFarmer/rsikit/blob/main/NOTICE).
