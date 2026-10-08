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
    """Base class for a policy with an asynchronous episode lifecycle.

    Generated source must define a top-level ``Solution`` subclass. An evaluator
    calls ``reset`` once and then ``act`` for each observation. Direct evaluation
    leaves cleanup to the caller; an executor closes its private policy copy.

    Args:
        observation_space: Gymnasium space describing observations.
        action_space: Gymnasium space describing valid actions.
        instructions: Task text available to the policy throughout its lifetime.

    Attributes:
        observation_space: Observation space supplied at construction.
        action_space: Action space supplied at construction.
        instructions: Task text supplied at construction.
        rng: NumPy generator initialized by ``reset``.
    """

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
        """Initialize the random generator and seed the action space.

        Overrides should await ``super().reset(seed=seed)`` and clear any state
        retained from previous episodes.

        Args:
            seed: Seed passed to NumPy and the action space; ``None`` uses fresh
                randomness.
        """
        self.rng = np.random.default_rng(seed)
        self.action_space.seed(seed)

    @abstractmethod
    async def act(self, observation: Observation) -> Action:
        """Choose an action from the current observation.

        Args:
            observation: A copy of the current environment observation.

        Returns:
            An action contained in ``action_space``.
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Release resources owned by the policy; the default does nothing.

        The executor calls this on its worker copy after an episode. Callers of
        ``evaluate`` must arrange cleanup themselves, including after failure.
        """


class PolicyDefinition(BaseModel):
    """An immutable source document; invalid Python can be retained for repair.

    Construction validates fields. Call ``validate()`` to check the Python
    contract. Neither loading nor saving imports or executes the source.

    Attributes:
        source: Exact Python source preserved without normalization; malformed
            Python is allowed until validation.
        name: Nonblank display name, defaulting to ``Solution``. This does not
            change the required class name, which is always ``Solution``.
        description: Human-readable description, excluded from the policy ID.

    Raises:
        pydantic.ValidationError: Fields have invalid types or unexpected names,
            or ``name`` is blank.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    source: str
    name: str = Field(default="Solution", pattern=r"\S")
    description: str = ""
    # Transient correlation for generation/evaluation progress; never serialized.
    _progress_attempt: str = PrivateAttr()

    @property
    def id(self) -> str:
        """SHA-256 of the name, a NUL separator, and exact UTF-8 source.

        Changing the description preserves identity; changing whitespace in the
        source or changing the name produces a different ID.
        """
        return hashlib.sha256((self.name + "\0" + self.source).encode()).hexdigest()

    @classmethod
    def from_text(
        cls, text: str, *, name: str | None = None, description: str | None = None
    ) -> "PolicyDefinition":
        """Read raw Python or Python with an rsikit metadata header.

        Args:
            text: Python source, optionally produced by ``to_text``.
            name: Display name override; ``None`` retains saved/default metadata.
            description: Description override; ``None`` retains saved/default metadata.

        Returns:
            An immutable definition without executing or statically validating it.

        Raises:
            ValueError: Text or serialized metadata is malformed.
            pydantic.ValidationError: Definition fields are invalid.
        """
        return PolicyEncoder.decode(text, name=name, description=description)

    @classmethod
    def from_file(
        cls, path: str | Path, *, name: str | None = None, description: str | None = None
    ) -> "PolicyDefinition":
        """Load a UTF-8 definition without executing its source.

        Args:
            path: Python file to read. Original line endings are preserved.
            name: Optional override for saved/default display name.
            description: Optional override for saved/default description.

        Returns:
            The decoded policy definition.

        Raises:
            OSError: The file cannot be read.
            ValueError: Metadata or definition fields are invalid.
        """
        with Path(path).open(encoding="utf-8", newline="") as stream:
            return cls.from_text(stream.read(), name=name, description=description)

    def to_text(self) -> str:
        """Return Python source prefixed with one JSON metadata comment.

        No validation or execution occurs. Decoding the result preserves source,
        metadata, and policy identity.
        """
        return PolicyEncoder.encode(self)

    def to_file(self, path: str | Path) -> None:
        """Atomically replace a file with the serialized UTF-8 definition.

        Args:
            path: Destination file. Its parent directory must already exist.

        Raises:
            OSError: Writing the temporary file or replacing the destination fails.
        """
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
        """Check syntax and the static ``Solution`` constructor contract.

        This checks neither inheritance nor runtime behavior; those are checked
        when the source executes.

        Raises:
            InvalidPolicy: Syntax, the top-level class, or constructor arguments
                violate the static contract.
        """
        PolicyEncoder.validate(self.source)


class InvalidPolicy(ValueError):
    """Policy source failed validation."""


class PolicyEncoder:
    """Serialize source documents and check their static Python contract."""

    @staticmethod
    def encode(definition: PolicyDefinition) -> str:
        """Serialize a definition without executing or validating its source.

        Args:
            definition: Policy source and metadata to serialize.

        Returns:
            Python text with a leading ``# rsikit-policy:`` JSON metadata comment.
        """
        metadata = json.dumps({"name": definition.name, "description": definition.description})
        return f"# rsikit-policy: {metadata}\n{definition.source}"

    @staticmethod
    def decode(
        text: str, *, name: str | None = None, description: str | None = None
    ) -> PolicyDefinition:
        """Decode raw Python or a single leading rsikit metadata comment.

        Args:
            text: Raw or serialized source; decoding never executes it.
            name: Optional override for the saved/default name.
            description: Optional override for the saved/default description.

        Returns:
            A definition preserving the source after the metadata header.

        Raises:
            ValueError: Input is not text, JSON is malformed, or metadata has
                unexpected fields or a repeated header.
            pydantic.ValidationError: Source or metadata fields are invalid.
        """
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
        """Check Python syntax and the statically visible constructor signature.

        Args:
            source: Python defining exactly one top-level ``Solution`` class. An
                explicit constructor must be synchronous and accept the worker's
                ``(observation_space, action_space, instructions="")`` call.

        Raises:
            InvalidPolicy: Syntax, class count, or constructor signature is invalid.

        Note:
            This does not execute source, check inheritance, or validate actions.
        """
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
