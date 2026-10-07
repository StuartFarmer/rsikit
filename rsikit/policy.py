"""Runtime policies and their non-executable source definitions."""

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
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

Observation = TypeVar("Observation")
Action = TypeVar("Action")


class Policy(ABC, Generic[Observation, Action]):
    """The runtime episode lifecycle implemented by a Solution."""

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


class PolicyDefinition(BaseModel):
    """An immutable source document; invalid Python can be retained for repair.

    Construction validates fields. Call validate() to check the Python contract.
    Neither loading nor saving imports or executes the source.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    source: str
    name: str = Field(default="Solution", pattern=r"\S")
    description: str = ""
    # Transient correlation for generation/evaluation progress; never serialized.
    _progress_attempt: str = PrivateAttr()

    @property
    def id(self) -> str:
        """Identity includes the name and exact source, but not the description."""
        return hashlib.sha256((self.name + "\0" + self.source).encode()).hexdigest()

    @classmethod
    def from_text(
        cls, text: str, *, name: str | None = None, description: str | None = None
    ) -> "PolicyDefinition":
        """Decode raw or serialized Python; explicit metadata overrides saved metadata."""
        return PolicyEncoder.decode(text, name=name, description=description)

    @classmethod
    def from_file(
        cls, path: str | Path, *, name: str | None = None, description: str | None = None
    ) -> "PolicyDefinition":
        """Load a UTF-8 Python solution; preserve its source and saved identity."""
        with Path(path).open(encoding="utf-8", newline="") as stream:
            return cls.from_text(stream.read(), name=name, description=description)

    def to_text(self) -> str:
        """Serialize source and metadata without validating or executing the source."""
        return PolicyEncoder.encode(self)

    def to_file(self, path: str | Path) -> None:
        """Atomically save a UTF-8 Python solution, including its identity metadata."""
        text, path = self.to_text(), Path(path)
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

    def validate(self) -> None:
        """Check source syntax and construction without executing it."""
        PolicyEncoder.validate(self.source)


class InvalidPolicy(ValueError):
    """Policy source failed validation."""


class PolicyEncoder:
    """Serialize source documents and check their static Python contract."""

    @staticmethod
    def encode(definition: PolicyDefinition) -> str:
        metadata = json.dumps({"name": definition.name, "description": definition.description})
        return f"# rsikit-policy: {metadata}\n{definition.source}"

    @staticmethod
    def decode(
        text: str, *, name: str | None = None, description: str | None = None
    ) -> PolicyDefinition:
        metadata = {}
        prefix = "# rsikit-policy: "
        if not isinstance(text, str):
            raise ValueError("Policy source must be text")
        if text.startswith(prefix):
            header, _, text = text.partition("\n")
            metadata = json.loads(header[len(prefix) :])
            if (
                not isinstance(metadata, dict)
                or metadata.keys() != {"name", "description"}
                or text.startswith(prefix)
            ):
                raise ValueError("Invalid policy metadata")
        definition = PolicyDefinition(source=text, **metadata)
        if name is None and description is None:
            return definition
        return PolicyDefinition(
            source=text,
            name=definition.name if name is None else name,
            description=definition.description if description is None else description,
        )

    @staticmethod
    def validate(source: str) -> None:
        """Check syntax and constructor arguments; behavior is checked during execution."""
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
