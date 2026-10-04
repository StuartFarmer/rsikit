"""Build pinned upstream Ocean bindings with the episodic-evaluation fix.

Run with the project's Python after installing the `ocean` extra. The source
checkout stays inside that Python environment; no C sources are vendored here.
"""

import argparse
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

UPSTREAM = "3b5c6046bb8b46685d62d151720025507e3418c2"
EPISODIC_PATCH = "episodes-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", default=["g2048", "breakout"])
    parser.add_argument("--source", type=Path, default=Path(sys.prefix) / "share" / "rsikit-ocean")
    args = parser.parse_args()
    if any(not name.isidentifier() for name in args.envs):
        parser.error("Environment names must be Python identifiers")
    source = args.source.resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--single-branch",
                "--branch",
                "3.0",
                "https://github.com/PufferAI/PufferLib.git",
                str(source),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "sparse-checkout",
                "set",
                "--no-cone",
                "/*",
                "!/pufferlib/",
                "!/resources/",
                "!/docs/",
                "!/tests/",
                "/pufferlib/*.py",
                "/pufferlib/environments/",
                "/pufferlib/ocean/*.py",
                "/pufferlib/ocean/*.h",
                *[f"/pufferlib/ocean/{name}/" for name in args.envs],
            ],
            cwd=source,
            check=True,
        )
        subprocess.run(["git", "checkout", UPSTREAM], cwd=source, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != UPSTREAM:
        parser.error(f"Expected upstream {UPSTREAM}, found {revision}")
    subprocess.run(
        [
            "git",
            "sparse-checkout",
            "add",
            *[
                f"/pufferlib/ocean/{name}/"
                for name in sorted(set(args.envs) | {"g2048", "breakout"})
            ],
        ],
        cwd=source,
        check=True,
    )
    patch = Path(__file__).with_name("ocean-episodes.patch").resolve()
    applied = (
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)], cwd=source, capture_output=True
        ).returncode
        == 0
    )
    if not applied:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
    # Force rebuilding: distutils does not track transitive C header changes.
    environment = dict(os.environ, NO_TRAIN="1")
    for name in args.envs:
        subprocess.run(
            [sys.executable, "setup.py", f"build_{name}", "--inplace", "--force"],
            cwd=source,
            env=environment,
            check=True,
        )
    (source / "pufferlib" / "rsikit_ocean.json").write_text(
        json.dumps(dict(upstream=UPSTREAM, episodic_patch=EPISODIC_PATCH)) + "\n"
    )
    (Path(sysconfig.get_path("purelib")) / "rsikit_ocean.pth").write_text(str(source) + "\n")
    # PufferLib creates this link on import. Precreate it for read-only Docker runs.
    resources = Path.cwd() / "resources"
    if not os.path.lexists(resources):
        resources.symlink_to(source / "pufferlib" / "resources", target_is_directory=True)
    print(f"Installed upstream Ocean {', '.join(args.envs)} from {UPSTREAM}")


if __name__ == "__main__":
    main()
