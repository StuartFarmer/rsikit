"""The executable policy contract; training is not part of this lifecycle."""

import ast
import hashlib
import json
import tempfile
from abc import ABC, abstractmethod
from inspect import Parameter, Signature
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from gymnasium import Space

Observation = TypeVar("Observation")
Action = TypeVar("Action")


class Policy(ABC, Generic[Observation, Action]):
    """A solution definition, with source persistence and an episode lifecycle."""

    @classmethod
    def from_text(
        cls, text: str, *, name: str | None = None, description: str | None = None
    ) -> type["Policy"]:
        """Load raw or serialized Python without importing or executing it.

        Serialized metadata supplies the name/description unless overridden.
        Raw Python defaults to name='Solution' and an empty description.
        The ID hashes the name and exact source; description edits do not change it.
        Loading does not validate source; optimizers use validate_policy explicitly.
        Execution creates a fresh instance from the returned definition in a fresh evaluation process.
        """
        metadata = {}
        prefix = "# rsikit-policy: "
        if text.startswith(prefix):
            header, _, text = text.partition("\n")
            metadata = json.loads(header[len(prefix) :])
            if (
                not isinstance(metadata, dict)
                or metadata.keys() != {"name", "description"}
                or not all(isinstance(value, str) for value in metadata.values())
                or not metadata["name"].strip()
                or text.startswith(prefix)
            ):
                raise ValueError("Invalid policy metadata")
        name = metadata.get("name", "Solution") if name is None else name
        description = metadata.get("description", "") if description is None else description
        if not isinstance(name, str) or not name.strip():
            raise ValueError("The policy needs a nonempty name")
        if not isinstance(description, str):
            raise ValueError("The policy description must be text")
        return type(
            "Solution",
            (cls,),
            {
                "name": name,
                "description": description,
                "id": hashlib.sha256((name + "\0" + text).encode()).hexdigest(),
                "_implementation": text,
            },
        )

    @classmethod
    def from_file(
        cls, path: str | Path, *, name: str | None = None, description: str | None = None
    ) -> type["Policy"]:
        """Load a UTF-8 Python solution; preserve its source and saved identity."""
        with Path(path).open(encoding="utf-8", newline="") as stream:
            return cls.from_text(stream.read(), name=name, description=description)

    @classmethod
    def to_text(cls) -> str:
        """Serialize source, name and description as Python with a metadata comment."""
        metadata = json.dumps({"name": cls.name, "description": cls.description})
        return f"# rsikit-policy: {metadata}\n{cls._implementation}"

    @classmethod
    def to_file(cls, path: str | Path) -> None:
        """Atomically save a UTF-8 Python solution, including its identity metadata."""
        text, path = cls.to_text(), Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(text)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def __init__(
        self,
        observation_space: Space[Observation],
        action_space: Space[Action],
        *,
        instructions: str = "",
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.instructions = instructions

    async def reset(self, *, seed: int | None = None) -> None:
        """Reset randomness; subclasses also clear their own episode memory."""
        self.rng = np.random.default_rng(seed)
        self.action_space.seed(seed)

    @abstractmethod
    async def act(self, observation: Observation) -> Action:
        """Return an action without access to the environment."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release policy resources."""


class InvalidPolicy(ValueError):
    """Policy source failed validation."""


def validate_policy(policy: type[Policy]) -> None:
    """Check policy source syntax and construction without importing or executing it.

    Optimizers call this after generation, before accepting a candidate. Runtime
    behavior is still checked by the executor; mutation rules belong to optimizers.
    """
    source = policy._implementation
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise InvalidPolicy(f"Invalid Python: {exc}") from exc
    solutions = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Solution"
    ]
    if len(solutions) != 1:
        raise InvalidPolicy("Policy source must define exactly one top-level Solution class")
    for method in reversed(solutions[0].body):
        if (
            not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            or method.name != "__init__"
        ):
            continue
        if isinstance(method, ast.AsyncFunctionDef):
            raise InvalidPolicy(
                "Solution.__init__ must be synchronous; initialize state in async reset"
            )
        args = method.args
        positional = [*args.posonlyargs, *args.args]
        required = len(positional) - len(args.defaults)
        parameters = [
            Parameter(
                arg.arg,
                Parameter.POSITIONAL_ONLY
                if i < len(args.posonlyargs)
                else Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty if i < required else None,
            )
            for i, arg in enumerate(positional)
        ]
        if args.vararg is not None:
            parameters.append(Parameter(args.vararg.arg, Parameter.VAR_POSITIONAL))
        parameters.extend(
            Parameter(
                arg.arg,
                Parameter.KEYWORD_ONLY,
                default=Parameter.empty if default is None else None,
            )
            for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        )
        if args.kwarg is not None:
            parameters.append(Parameter(args.kwarg.arg, Parameter.VAR_KEYWORD))
        try:
            # Bind the worker's actual call without evaluating any generated code or defaults.
            Signature(parameters).bind(None, None, None, instructions="")
        except (TypeError, ValueError) as exc:
            raise InvalidPolicy(
                "Solution.__init__ must accept (self, observation_space, action_space, *, "
                "instructions=''). Prefer removing __init__ and initializing state in "
                f"async reset after await super().reset(seed=seed). Signature mismatch: {exc}"
            ) from exc
        break  # Python uses the last definition of a method in the class body.


MAX_SOURCE = 65_536


def load_policy(source, observation_space, action_space, instructions):
    """Load a Solution instance in an evaluation process."""
    if len(source.encode()) > MAX_SOURCE:
        raise ValueError("Source exceeds 64 KiB")
    namespace = {"__name__": "candidate"}
    exec(compile(source, "candidate.py", "exec"), namespace)
    solution = namespace["Solution"]
    if not isinstance(solution, type) or not issubclass(solution, Policy):
        raise TypeError("Solution must subclass rsikit.Policy")
    return solution(observation_space, action_space, instructions=instructions)
