"""Check release archives: python scripts/check_distribution.py DIST_DIRECTORY."""

import sys
import tarfile
import zipfile
from pathlib import Path


def main(directory):
    directory = Path(directory)
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, "Use a fresh directory with one wheel and sdist"
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel = set(archive.namelist())
    with tarfile.open(sources[0]) as archive:
        source = {name.partition("/")[2] for name in archive.getnames()}
    for name in (
        "rsikit/__init__.py",
        "research/cli.py",
        "rsikit/generation/prompts/generate_policy.j2",
        "rsikit/generation/prompts/libraries.txt",
        "research/elitesearch/prompts/new.j2",
        "rsikit/envs/data/bitcoin_train.csv",
        "rsikit/envs/data/README.md",
        "rsikit/envs/data/blackjack/License.txt",
        "rsikit/envs/fonts/GUST-FONT-LICENSE.TXT",
        "rsikit/envs/fonts/lmroman10-regular.otf",
        "research/alphaevolve/NOTICE",
        "research/evox/UPSTREAM_LICENSE",
        "research/ocean/LICENSE",
        "research/ocean/NOTICE",
    ):
        assert name in wheel and name in source, f"Missing runtime resource: {name}"
    for name in ("LICENSE", "NOTICE"):
        assert any(p.endswith(f".dist-info/licenses/{name}") for p in wheel), name
        assert name in source, name
    for name in (
        "book.toml",
        "docs/src/index.md",
        "docs/src/SUMMARY.md",
        "scripts/mdbook_api.py",
        "examples/existing_system.py",
        "examples/custom_system.py",
    ):
        assert name in source, f"Missing source-release input: {name}"
    for files in (wheel, source):
        for name in files:
            parts = Path(name).parts
            assert not any(
                p in parts for p in (".env", ".git", ".venv", "__pycache__", "maintainer")
            ), name
            assert not name.startswith(("data/forex/", "docs/superpowers/", "docs/research/")), name
            assert "VISUALIZATION_PROPOSAL" not in name, name
    print(f"Distribution contents OK: {wheels[0].name}, {sources[0].name}")


if __name__ == "__main__":
    main(sys.argv[1])
