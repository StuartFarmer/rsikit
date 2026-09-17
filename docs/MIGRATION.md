# Extraction from slick-bits

Extracted from https://github.com/StuartFarmer/slick-bits at `355335f290ddb1ea9b08dbd41aeccc6136779a69` on 2026-09-17.
The new repository retains filtered history for RSIKit, its research/design documents
and the shared provider helper. The extraction also includes all uncommitted RSIKit
concept-module work from that checkout.

The package remains `rsikit`; provider/model behavior and algorithms are unchanged.
The shared offline provider now lives in `rsikit/tests/providers.py`. Prompt/example
assets are packaged. Local donor links point to pinned source in slick-bits.
The development environment uses the independently installed Slick dependency.
No GitHub repository has been created and no commits have been pushed.

Verified: source checkout and isolated installed wheel each passed 62 tests; six
Docker tests were skipped. Wheel assets, offline examples, local documentation
links and package Ruff checks passed. No live model calls were made.
