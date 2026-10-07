"""CLI/config equivalence and errors before any experiment starts."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import yaml

BASE = dict(
    env="CartPole-v1",
    optimizer="elite",
    model="scripted",
    output="run",
)


class CliTests(unittest.TestCase):
    def test_all_builtin_selectors_variants_and_invalid_options(self):
        from research.cli import parse_config

        for selector, field in (
            ("alphaevolve", "proposals"),
            ("shinka", "generations"),
            ("elite", "population"),
            ("lineage", "families"),
        ):
            with self.subTest(selector=selector):
                config = parse_config(["run"], config={**BASE, "optimizer": selector})
                self.assertEqual(config["optimizer"], selector)
                self.assertEqual(parse_config(["run"], config=config), config)
                for invalid in ({field: -1}, {"typo": 1}, {"generation_concurrency": 0}):
                    with self.assertRaises(ValueError):
                        parse_config(
                            ["run"],
                            config={**BASE, "optimizer": selector, "optimizer_options": invalid},
                        )
                with self.assertRaises(ValueError):
                    parse_config(
                        ["run"], config={**BASE, "optimizer": selector, "videos": {"top": 1}}
                    )
        for variant in ("paper", "original", "improved"):
            config = parse_config(
                ["run", "--variant", variant], config={**BASE, "optimizer": "alphaevolve"}
            )
            self.assertEqual(config["optimizer_options"]["variant"], variant)

    def test_budget_is_opt_in(self):
        from research.cli import parse_config

        for options in ({}, {"budget": {}}, {"budget": {"max_calls": 2}}):
            with self.subTest(options=options):
                config = parse_config(["run"], config={**BASE, **options})
                self.assertIsNone(config["budget"]["spend_cap"])
                self.assertIsNone(config["budget"]["max_tokens"])
                self.assertEqual(
                    config["budget"]["max_calls"], options.get("budget", {}).get("max_calls")
                )
                self.assertEqual(parse_config(["run"], config=config), config)
        for budget in ({"spend_cap": 1}, {"max_calls": 0}, {"max_tokens": -1}):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                parse_config(["run"], config={**BASE, "budget": budget})

    def test_file_overrides_paths_and_roundtrip(self):
        from research.cli import parse_config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        **BASE,
                        "optimizer_options": {"population": 50, "max_repairs": 0},
                        "evaluation": {"workers": 2},
                        "selection": {"test_seeds": {"start": 100, "stop": 105}},
                    }
                )
            )
            config = parse_config(
                ["run", "--config", str(path), "--population", "100", "--test-seeds"]
            )
            self.assertEqual(config["optimizer_options"]["population"], 100)
            self.assertEqual(config["optimizer_options"]["max_repairs"], 0)
            self.assertEqual(config["evaluation"]["workers"], 2)
            self.assertEqual(config["selection"]["test_seeds"], [])
            self.assertEqual(config["output"], str((Path(directory) / "run").resolve()))
            path.write_text(yaml.safe_dump(config))
            self.assertEqual(parse_config(["run", "--config", str(path)]), config)
            changed = parse_config(["run", "--config", str(path), "--output", "new-run"])
            self.assertEqual(changed["output"], str(Path("new-run").resolve()))

    def test_invalid_config_and_flags(self):
        from research.cli import parse_config

        invalid = [
            {"typo": 1},
            {"version": 2},
            {"evaluation": {"workers": True}},
            {"evaluation": {"seeds": [False]}},
            {"evaluation": {"seeds": [1, 1]}},
            {"selection": {"test_seeds": [0]}},
            {"evaluation": {"batch_size": 32}},
            {"optimizer_options": {"new_fraction": 0.8, "remix_fraction": 0.8}},
            {"optimizer_options": {"population": 2, "elites": 3}},
            {"generation": {"timeout": float("inf")}},
            {"videos": {"top": 1}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.yaml"
            for changes in invalid:
                with self.subTest(changes=changes):
                    path.write_text(yaml.safe_dump({**BASE, **changes}))
                    with self.assertRaises((ValueError, SystemExit)):
                        parse_config(["run", "--config", str(path)])
            path.write_text('{"env":"Blackjack","env":"CartPole-v1"}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                parse_config(["run", "--config", str(path)])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_config(["run", "--optimizer", "elite", "--popul", "2"])

    def test_help_and_evaluation_need_no_model(self):
        from research.cli import parse_config

        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as error:
            parse_config(["run", "--env", "Blackjack", "--optimizer", "elite", "--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--population", out.getvalue())
        self.assertIn("--env-shoes-per-episode", out.getvalue())
        config = parse_config(
            ["evaluate", "--env", "CartPole-v1", "--policy", __file__, "--output", "out"]
        )
        self.assertNotIn("generation", config)
        self.assertNotIn("optimizer", config)

    def test_ocean_config_and_seed_limits_without_native_import(self):
        from research.cli import parse_config

        config = parse_config(
            ["evaluate", "--env", "ocean:g2048", "--policy", __file__, "--output", "out"]
        )
        self.assertEqual(config["evaluation"]["batch_size"], 32)
        self.assertEqual(config["evaluation"]["max_steps"], 2000)
        self.assertEqual(config["evaluation"]["score_key"], "merge_score")
        with self.assertRaises(ValueError):
            parse_config(
                [
                    "evaluate",
                    "--env",
                    "ocean:g2048",
                    "--policy",
                    __file__,
                    "--output",
                    "out",
                    "--seeds",
                    str(2**31),
                ]
            )

    def test_custom_modules_and_stale_options(self):
        from research.cli import load_component, parse_config

        source = """from pydantic import BaseModel, ConfigDict
class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    count: int = 1
def add_arguments(parser):
    parser.add_argument("--count", type=int)
async def optimize(**kwargs):
    return []
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("a", "b"):
                (root / folder).mkdir()
                (root / folder / "opt.py").write_text(source)
            first = load_component("a/opt.py", kind="optimizer", base_dir=root)
            second = load_component("b/opt.py", kind="optimizer", base_dir=root)
            self.assertNotEqual(first.__name__, second.__name__)
            path = root / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        **BASE,
                        "optimizer": "a/opt.py",
                        "optimizer_options": {"count": 3},
                    }
                )
            )
            config = parse_config(["run", "--config", str(path)])
            self.assertEqual(config["optimizer_options"], {"count": 3})
            with self.assertRaises(ValueError):
                parse_config(["run", "--config", str(path), "--optimizer", "elite"])

    def test_plain_help_and_boolean_version(self):
        from research.cli import parse_config

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            parse_config(["--help"])
        self.assertEqual(error.exception.code, 0)
        with self.assertRaises(ValueError):
            parse_config(["run"], config={**BASE, "version": True})

    def test_typed_environment_options_and_metric_validation(self):
        from research.cli import parse_config

        config = parse_config(
            [
                "evaluate",
                "--env",
                "ocean:g2048",
                "--env-reward-scaler",
                "2",
                "--env-use-sparse-reward",
                "--policy",
                __file__,
                "--output",
                "out",
            ]
        )
        self.assertEqual(config["environment"]["reward_scaler"], 2)
        self.assertTrue(config["environment"]["use_sparse_reward"])
        with self.assertRaises(ValueError):
            parse_config(
                [
                    "evaluate",
                    "--env",
                    "ocean:breakout",
                    "--score-key",
                    "merge_score",
                    "--policy",
                    __file__,
                    "--output",
                    "out",
                ]
            )

    def test_reserved_optimizer_fields_fail_preflight(self):
        from research.cli import load_component

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.py"
            path.write_text("""from pydantic import BaseModel, ConfigDict
class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generation: int = 7
def add_arguments(parser): pass
async def optimize(**kwargs): return []
""")
            with self.assertRaisesRegex(ValueError, "reserved"):
                load_component(str(path), kind="optimizer", base_dir=path.parent)

    def test_declared_paths_in_frozen_custom_options(self):
        from research.cli import parse_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "optimizer.py").write_text(
                """from pathlib import Path
from pydantic import BaseModel, ConfigDict
class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    data: Path
 def_placeholder
""".replace(
                    " def_placeholder",
                    """def add_arguments(parser): parser.add_argument("--data")
async def optimize(**kwargs): return []
""",
                )
            )
            config = root / "config.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        **BASE,
                        "optimizer": "optimizer.py",
                        "optimizer_options": {"data": "data.txt"},
                    }
                )
            )
            parsed = parse_config(["run", "--config", str(config)])
            self.assertEqual(
                parsed["optimizer_options"]["data"], str((root / "data.txt").resolve())
            )
            parsed = parse_config(["run", "--config", str(config), "--data", "override.txt"])
            self.assertEqual(
                parsed["optimizer_options"]["data"], str(Path("override.txt").resolve())
            )

    def test_yaml_config_comments_overrides_and_duplicate_keys(self):
        from research.cli import parse_config, read_config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text("""# Entire experiment, with a CLI override below.
env: CartPole-v1
optimizer: elite
model: scripted
output: run
optimizer_options:
  population: 50
  max_repairs: 0
budget:
  spend_cap: 100
  input_price: 0.01
  output_price: 0.01
evaluation:
  seeds: [7, 9]
selection:
  test_seeds: {start: 100, stop: 102}
""")
            parsed = parse_config(["run", "--config", str(path), "--population", "100"])
            self.assertEqual(parsed["optimizer_options"]["population"], 100)
            self.assertEqual(parsed["optimizer_options"]["max_repairs"], 0)
            self.assertEqual(parsed["selection"]["test_seeds"], [100, 101])
            path.write_text("environment: &options {value: 1}\noptimizer_options: *options\n")
            loaded = read_config(path)
            loaded["environment"]["value"] = 2
            self.assertEqual(loaded["optimizer_options"]["value"], 1)
            for content in (
                "evaluation:\n  workers: 1\n  workers: 2\n",
                "env: [invalid",
                "env: !!python/object/apply:os.system ['false']",
                "env: .nan",
                "env: 2026-10-03",
            ):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read_config(path)
