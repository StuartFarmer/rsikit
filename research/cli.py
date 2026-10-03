"""Run searches and evaluate policies using flags or a complete YAML config."""

import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from research.experiment import EvaluationConfig, Options


class Generation(Options):
    provider: Literal["openrouter"] = "openrouter"
    concurrency: int = Field(default=4, ge=1)
    timeout: float = Field(default=120, gt=0)
    max_input_tokens: int = Field(default=65536, ge=1)
    max_output_tokens: int = Field(default=16384, ge=1)


class Budget(Options):
    spend_cap: float = Field(gt=0)
    input_price: float = Field(gt=0)
    output_price: float = Field(gt=0)
    max_calls: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)


class Selection(Options):
    finalists: int = Field(default=1, ge=1)
    validation_seeds: list[int] = Field(default_factory=list)
    test_seeds: list[int] = Field(default_factory=list)


class Videos(Options):
    top: int = Field(default=0, ge=0)
    workers: int = Field(default=1, ge=1)


class Common(Options):
    version: Literal[1] = 1
    command: Literal["run", "evaluate"]
    env: str
    output: str
    environment: dict = Field(default_factory=dict)
    evaluation: dict = Field(default_factory=dict)


class SearchConfig(Common):
    optimizer: str
    model: str
    search_seed: int = Field(default=0, ge=0)
    optimizer_options: dict = Field(default_factory=dict)
    generation: dict = Field(default_factory=dict)
    budget: dict
    selection: dict = Field(default_factory=dict)
    videos: dict = Field(default_factory=dict)


class EvaluateConfig(Common):
    policies: list[str] = Field(min_length=1)


class _ConfigLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Config keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def read_config(path):
    try:
        value = yaml.load(Path(path).read_text(), Loader=_ConfigLoader)
        # Validate plain data and detach aliases so an override changes only its own field.
        value = json.loads(json.dumps(value, allow_nan=False))
    except (yaml.YAMLError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Config must be a YAML mapping")
    return value


def _selector(value, base):
    if value.endswith(".py"):
        return str((base / value).resolve())
    return value


def load_component(selector: str, *, kind: str, base_dir: Path):
    if selector.endswith(".py"):
        path = (base_dir / selector).resolve()
        if not path.is_file():
            raise ValueError(f"{kind} file does not exist: {path}")
        name = "_rsikit_" + hashlib.sha256(str(path).encode()).hexdigest()
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        if kind == "environment":
            if not hasattr(module, "environment"):
                raise ValueError(f"{path} must export environment")
            component = module.environment
        else:
            component = module
    elif kind == "optimizer":
        if selector != "elite":
            raise ValueError(f"Unknown optimizer {selector!r}; choose elite or a .py file")
        component = importlib.import_module("research.elitesearch.cli")
    elif selector.startswith("ocean:"):
        from research.ocean.environment import definition

        component = definition(selector.removeprefix("ocean:"))
    else:
        from research.environments import definition

        component = definition(selector)
    required = (
        ("Options", "add_arguments", "optimize")
        if kind == "optimizer"
        else (
            "Options",
            "add_arguments",
            "describe",
            "open_evaluator",
            "evaluation_defaults",
            "supported_evaluation_fields",
            "score_keys",
            "protocol",
            "provenance",
        )
    )
    for name in required:
        if not hasattr(component, name):
            raise ValueError(f"{kind} {selector!r} must export {name}")
    if not isinstance(component.Options, type) or not issubclass(component.Options, BaseModel):
        raise ValueError(f"{kind}.Options must be a Pydantic model")
    if kind == "optimizer" and {"generation", "videos"}.intersection(
        component.Options.model_fields
    ):
        raise ValueError("Optimizer options generation and videos are reserved runtime fields")
    if component.Options.model_config.get("extra") != "forbid":
        raise ValueError(f"{kind}.Options must forbid extra fields")
    return component


def _validate(model, value, prefix=""):
    try:
        # JSON strict validation accepts declared Path fields while rejecting numeric strings.
        result = model.model_validate_json(json.dumps(value, allow_nan=False), strict=True)
    except ValidationError as exc:
        raise ValueError(
            "; ".join(
                f"{'.'.join(filter(None, [prefix, *map(str, e['loc'])]))}: {e['msg']}"
                for e in exc.errors()
            )
        ) from exc
    return result


def _resolve_options(model, values, base, overrides):
    result = _validate(model, values)

    def resolve(value, origin):
        if isinstance(value, Path):
            return (origin / value).resolve()
        if isinstance(value, BaseModel):
            return value.model_copy(
                update={
                    key: resolve(getattr(value, key), origin) for key in type(value).model_fields
                }
            )
        elif isinstance(value, list):
            return [resolve(v, origin) for v in value]
        elif isinstance(value, dict):
            return {k: resolve(v, origin) for k, v in value.items()}
        return value

    result = result.model_copy(
        update={
            key: resolve(getattr(result, key), Path.cwd() if key in overrides else base)
            for key in type(result).model_fields
        }
    )
    return result.model_dump(mode="json")


def _seeds(value, path, *, empty=False):
    if isinstance(value, dict):
        if set(value) != {"start", "stop"} or any(type(v) is not int for v in value.values()):
            raise ValueError(f"{path}: expected integer start/stop")
        if value["start"] < 0 or value["stop"] <= value["start"]:
            raise ValueError(f"{path}: invalid seed range")
        value = list(range(value["start"], value["stop"]))
    if not isinstance(value, list) or any(type(v) is not int or v < 0 for v in value):
        raise ValueError(f"{path}: expected nonnegative integer seeds")
    if (not empty and not value) or len(set(value)) != len(value):
        raise ValueError(
            f"{path}: seeds must be unique and {'may be empty' if empty else 'nonempty'}"
        )
    return value


def _register(parser, component, *, environment=False):
    before = set(parser._actions)
    component.add_arguments(parser)
    added = [a for a in parser._actions if a not in before]
    existing = {a.dest for a in before}
    for action in added:
        if action.dest in existing:
            raise ValueError(f"Component option conflicts with existing field: {action.dest}")
        existing.add(action.dest)
        if environment and any(
            not flag.startswith(("--env-", "--no-env-")) for flag in action.option_strings
        ):
            raise ValueError("Environment flags must start with --env-")
        if action.dest not in component.Options.model_fields:
            raise ValueError(f"Option {action.dest} has no corresponding Options field")
        action.default = argparse.SUPPRESS
        # Requiredness belongs to merged validation, so files can supply required options.
        action.required = False
    return {a.dest for a in added}


def parse_config(argv=None, *, config=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv in (["--help"], ["-h"]):
        help_parser = argparse.ArgumentParser(prog="rsikit", description=__doc__)
        help_parser.add_argument("command", choices=("run", "evaluate"))
        help_parser.print_help()
        raise SystemExit(0)
    bootstrap = argparse.ArgumentParser(
        add_help=False, allow_abbrev=False, argument_default=argparse.SUPPRESS
    )
    bootstrap.add_argument("command", choices=("run", "evaluate"))
    for key in ("config", "env", "optimizer"):
        bootstrap.add_argument("--" + key)
    selectors, _ = bootstrap.parse_known_args(argv)
    raw = vars(selectors)
    base = Path(raw["config"]).resolve().parent if "config" in raw else Path.cwd()
    data = (
        deepcopy(config)
        if config is not None
        else read_config(raw["config"])
        if "config" in raw
        else {}
    )
    if "command" in data and data["command"] != raw["command"]:
        raise ValueError("Config command differs from requested command")
    data["command"] = raw["command"]
    for key in ("env", "optimizer"):
        if key in raw:
            data[key] = _selector(raw[key], Path.cwd())
        elif key in data:
            if not isinstance(data[key], str):
                raise ValueError(f"{key}: expected a string")
            data[key] = _selector(data[key], base)
    env = load_component(data["env"], kind="environment", base_dir=base) if "env" in data else None
    optimizer = (
        load_component(data["optimizer"], kind="optimizer", base_dir=base)
        if "optimizer" in data
        else None
    )
    parser = argparse.ArgumentParser(
        prog="rsikit",
        parents=[bootstrap],
        allow_abbrev=False,
        argument_default=argparse.SUPPRESS,
        description=__doc__,
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Validate and print without running"
    )
    destinations = {}

    def option(flag, section=None, key=None, **kwargs):
        dest = flag.replace("-", "_")
        parser.add_argument("--" + flag, **kwargs)
        destinations[dest] = (section, key or dest)

    option("output")
    for flag, key, kind in (
        ("workers", "workers", int),
        ("timeout-per-seed", "timeout_per_seed", float),
        ("batch-size", "batch_size", int),
        ("max-steps", "max_steps", int),
        ("score-key", "score_key", str),
    ):
        option(flag, "evaluation", key, type=kind)
    option("seeds", "evaluation", type=int, nargs="+")
    if raw["command"] == "run":
        option("model")
        option("search-seed", type=int)
        option("provider", "generation")
        option("generation-concurrency", "generation", "concurrency", type=int)
        option("generation-timeout", "generation", "timeout", type=float)
        for flag in ("max-input-tokens", "max-output-tokens"):
            option(flag, "generation", type=int)
        for flag in ("spend-cap", "input-price", "output-price", "max-calls", "max-tokens"):
            option(flag, "budget", type=int if flag in ("max-calls", "max-tokens") else float)
        option("finalists", "selection", type=int)
        option("validation-seeds", "selection", type=int, nargs="*")
        option("test-seeds", "selection", type=int, nargs="*")
        option("video-top", "videos", "top", type=int)
        option("video-workers", "videos", "workers", type=int)
    else:
        option("policy", key="policies", nargs="+")
    env_fields = _register(parser, env, environment=True) if env else set()
    opt_fields = _register(parser, optimizer) if optimizer else set()
    flags = vars(parser.parse_args(argv))
    for key, value in flags.items():
        if key in destinations:
            section, field = destinations[key]
            if section:
                if not isinstance(data.setdefault(section, {}), dict):
                    raise ValueError(f"{section}: expected an object")
                data[section][field] = value
            else:
                data[field] = value
        elif key in env_fields | opt_fields:
            section = "environment" if key in env_fields else "optimizer_options"
            if not isinstance(data.setdefault(section, {}), dict):
                raise ValueError(f"{section}: expected an object")
            data[section][key] = value
    if type(data.get("version", 1)) is not int:
        raise ValueError("version: expected integer 1")
    model = SearchConfig if raw["command"] == "run" else EvaluateConfig
    config = _validate(model, data).model_dump()
    for key in ("env", "output", *(("optimizer", "model") if config["command"] == "run" else ())):
        if not config[key].strip():
            raise ValueError(f"{key}: must not be empty")
    config["output"] = str(
        ((Path.cwd() if "output" in flags else base) / config["output"]).resolve()
    )
    if "policies" in config:
        config["policies"] = [
            str(((Path.cwd() if "policy" in flags else base) / p).resolve())
            for p in config["policies"]
        ]
    config["environment"] = _resolve_options(
        env.Options, config["environment"], base, flags.keys() & env_fields
    )
    evaluation = {**env.evaluation_defaults, **config["evaluation"]}
    evaluation["seeds"] = _seeds(evaluation.get("seeds", list(range(10))), "evaluation.seeds")
    evaluation = _validate(EvaluationConfig, evaluation, "evaluation").model_dump()
    for key in ("batch_size", "max_steps", "score_key"):
        if key not in env.supported_evaluation_fields and evaluation[key] is not None:
            raise ValueError(f"evaluation.{key}: unsupported by {config['env']}")
    if evaluation["score_key"] not in env.score_keys:
        raise ValueError(f"evaluation.score_key: choose from {env.score_keys}")
    config["evaluation"] = evaluation
    panels = [evaluation["seeds"]]
    if config["command"] == "run":
        config["optimizer_options"] = _resolve_options(
            optimizer.Options, config["optimizer_options"], base, flags.keys() & opt_fields
        )
        config["generation"] = _validate(
            Generation, config["generation"], "generation"
        ).model_dump()
        budget = _validate(Budget, config["budget"], "budget").model_dump()
        if budget["max_calls"] is None:
            if config["optimizer"] != "elite":
                raise ValueError("budget.max_calls is required for custom optimizers")
            opt = config["optimizer_options"]
            budget["max_calls"] = opt["population"] * opt["generations"] * (1 + opt["max_repairs"])
        if budget["max_tokens"] is None:
            budget["max_tokens"] = budget["max_calls"] * sum(
                config["generation"][k] for k in ("max_input_tokens", "max_output_tokens")
            )
        config["budget"] = budget
        selection = config["selection"]
        for key in ("validation_seeds", "test_seeds"):
            selection[key] = _seeds(selection.get(key, []), "selection." + key, empty=True)
            panels.append(selection[key])
        config["selection"] = _validate(Selection, selection, "selection").model_dump()
        config["videos"] = _validate(Videos, config["videos"], "videos").model_dump()
        if config["videos"]["top"] and (
            config["env"] != "Blackjack"
            or config["optimizer"] != "elite"
            or evaluation["max_steps"] is not None
        ):
            raise ValueError("videos.top requires full-episode Blackjack with elite")
    seen = set()
    for seeds in panels:
        if seen.intersection(seeds):
            raise ValueError("Search, validation and test seeds must be disjoint")
        seen.update(seeds)
    if env.protocol == "ocean-upstream-fixed-horizon-v1":
        width, horizon = evaluation["batch_size"], evaluation["max_steps"]
        if width is None or horizon is None or max(width, horizon) > 2**31 - 1:
            raise ValueError("Ocean requires positive int32 batch_size and max_steps")
        if any((s + 1) * width > 2**31 for s in seen):
            raise ValueError("Ocean seeds and batch_size exceed upstream signed integer limits")
    return config


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        config = parse_config(argv)
        if "--print-config" in argv:
            print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), end="")
            return
        from research.experiment import evaluate_policies, run_experiment

        summary = asyncio.run(
            (run_experiment if config["command"] == "run" else evaluate_policies)(config)
        )
        print(json.dumps(summary, indent=2, allow_nan=False))
        if summary["status"] not in ("completed", "budget_exhausted"):
            raise SystemExit(1)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"rsikit: {exc}") from exc


if __name__ == "__main__":
    main()
