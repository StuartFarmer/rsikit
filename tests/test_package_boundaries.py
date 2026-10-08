"""Research algorithms are independent clients of the common library."""

import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {
    "alphaevolve",
    "shinkaevolve",
    "elitesearch",
    "lineagesearch",
    "reevo",
    "evox",
    "gepa",
    "eoh",
    "promptbreeder",
    "stop_optimizer",
}


class PackageBoundariesTests(unittest.TestCase):
    def test_research_and_core_import_boundaries(self):
        packages = ["rsikit", *(f"research/{name}" for name in sorted(ALGORITHMS))]
        for package in packages:
            directory = ROOT / package
            self.assertTrue(directory.is_dir(), f"Missing package: {package}")
            for path in directory.rglob("*.py"):
                parts = path.relative_to(ROOT).with_suffix("").parts
                parent = ".".join(parts[:-1])
                for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if node.level:
                            module = importlib.util.resolve_name("." * node.level + module, parent)
                        names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
                    else:
                        continue
                    for name in names:
                        top = name.split(".")[0]
                        forbidden = top in ALGORITHMS | {"examples", "tests"}
                        if top == "research":
                            forbidden |= package == "rsikit" or not (
                                name == "research"
                                or (path.name == "cli.py" and name.startswith("research.providers"))
                                or name == package.replace("/", ".")
                                or name.startswith(package.replace("/", ".") + ".")
                            )
                        self.assertFalse(
                            forbidden, f"{path.relative_to(ROOT)}:{node.lineno}: {name}"
                        )


if __name__ == "__main__":
    unittest.main()
