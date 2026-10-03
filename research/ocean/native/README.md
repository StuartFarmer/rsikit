# Pinned native 2048 source

`g2048.h`, `pufferenv.h`, `ini.h`, and `LICENSE` are unmodified files from
[PufferLib commit 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2](https://github.com/PufferAI/PufferLib/tree/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2)
(respectively `ocean/g2048/g2048.h`, `src/pufferenv.h`, `src/ini.h`, `LICENSE`).
PufferLib is MIT licensed; its license is retained here.

`raylib.h` is our minimal no-op rendering/input shim. It always reports that no
window is open; native gameplay therefore follows the ordinary action path.
`adapter.c` owns fresh zero-initialized native games, disables scaffolding, and
sets each game's RNG to its requested seed before its first reset. No upstream
source or mechanics are patched. Native terminal results are captured once from
the completed log. Only on termination, the last transition is replayed on a
copy through upstream's `step_without_reset` to classify game-over versus timeout.
This does not touch the live game or its reset observation.

Build requires a POSIX host and C compiler (`CC`, default `cc`). The Python
adapter hashes source, compiler identity, flags and platform into a temporary
shared-library cache, and performs no network access. RNG streams use native
`rand_r`; reproducibility is platform-specific. Compiled binaries are not vendored.
