"""The executable policy contract; training is not part of this lifecycle."""

import hashlib
import json
import tempfile
from abc import ABC, abstractmethod
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
        Execution creates a fresh instance from the returned definition in its sandbox.
        """
        # Generation imports Policy, so load its validator only when called.
        from .generation.edits import check_program

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
        check_program(text)
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
