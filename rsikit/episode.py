"""Episode trajectories returned by Evaluator."""

import base64
import math
from typing import Annotated, Union

import numpy as np
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field
from typing_extensions import TypeAliasType


def _reward(value):
    # Strict float validation also accepts float-convertible objects, including NumPy booleans.
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("Episode rewards must be finite ordinary numbers")
    return value


def _array(value):
    if value.dtype.kind not in "biuf" or value.dtype.itemsize > 16:
        raise ValueError("Episode arrays must contain real numbers or booleans")
    if value.nbytes > 262_144:
        raise ValueError("Array exceeds 256 KiB")
    return value


EpisodeValue = TypeAliasType(
    "EpisodeValue",
    Union[
        Annotated[np.ndarray, AfterValidator(_array)],
        np.integer,
        np.floating,
        np.bool_,
        None,
        bool,
        int,
        float,
        str,
        bytes,
        list["EpisodeValue"],
        tuple["EpisodeValue", ...],
        dict[str, "EpisodeValue"],
    ],
)


class Episode(BaseModel):
    """An attempted rollout, including its partial trajectory and candidate error.

    Recorded transitions have T actions/rewards/flags and T+1 observations/infos.
    Initialization failures may have no initial observation. An error means the
    accumulated rewards are partial evidence, not a successful fitness score.

    Transition t is observations[t], actions[t], rewards[t], observations[t+1],
    terminations[t], truncations[t], infos[t+1]. Index 0 of infos is reset info.
    ``len(episode)`` is the number of recorded rewards. Fields are mutable; call
    ``validate_complete`` before trusting a manually assembled trajectory.

    Attributes:
        observations: Initial observation followed by one observation per transition.
        actions: Actions actually applied to the environment.
        rewards: Finite ordinary integer or float rewards, one per transition.
        terminations: Environment termination flag for each transition.
        truncations: Environment or evaluator truncation flag for each transition.
        infos: Reset info followed by each transition's info.
        artifacts: Relative file names mapped to raw bytes, populated by workers
            or supplied by callers.
        error: Nonblank failure diagnostic, or ``None`` for a successful episode.

    Note:
        Nested values support native containers, NumPy scalars, and real or
        boolean NumPy arrays of at most 256 KiB each. ``encode`` preserves native
        values for pickle; it does not produce a JSON-safe dictionary.
    """

    model_config = ConfigDict(
        strict=True, extra="forbid", arbitrary_types_allowed=True, revalidate_instances="always"
    )

    observations: list[EpisodeValue] = Field(default_factory=list)
    actions: list[EpisodeValue] = Field(default_factory=list)
    rewards: list[Annotated[int | float, BeforeValidator(_reward)]] = Field(default_factory=list)
    terminations: list[bool] = Field(default_factory=list)
    truncations: list[bool] = Field(default_factory=list)
    infos: list[dict[str, EpisodeValue]] = Field(default_factory=list)
    artifacts: dict[str, bytes] = Field(default_factory=dict)
    error: Annotated[str, Field(pattern=r"\S")] | None = None

    def validate_complete(self) -> "Episode":
        """Validate current fields and transition alignment after any mutations.

        Successful episodes must have at least one transition and end exactly
        at the final transition. Failed episodes may contain a partial trajectory
        or no observations when initialization failed.

        Returns:
            This episode, unchanged.

        Raises:
            ValueError: Fields, lengths, or termination/truncation positions are invalid.
        """
        type(self).model_validate(self)
        length = len(self)
        if (not length and self.error is None) or any(
            len(track) != length for track in (self.actions, self.terminations, self.truncations)
        ):
            raise ValueError("Episode transitions are not aligned")
        initial = len(self.observations)
        if initial != len(self.infos) or (
            initial != length + 1 and not (self.error is not None and length == initial == 0)
        ):
            raise ValueError("Episode must include its initial observation and info")
        ends = list(zip(self.terminations, self.truncations))
        if any(a or b for a, b in ends[:-1]) or (self.error is None and not any(ends[-1])):
            raise ValueError("Episode must end exactly at its last transition")
        return self

    def encode(self) -> dict:
        """Return validated native values for pickle transport and storage.

        Returns:
            All trajectory and artifact fields; ``error`` is omitted when unset.

        Raises:
            ValueError: ``validate_complete`` rejects the episode.
        """
        self.validate_complete()
        return self.model_dump(exclude={"error"} if self.error is None else set())

    @classmethod
    def from_data(cls, data: dict) -> "Episode":
        """Construct an episode from native fields or legacy tagged JSON values.

        Args:
            data: Dictionary produced by ``encode`` or the legacy encoder.

        Returns:
            A validated, complete successful or failed episode.

        Raises:
            ValueError: Field names, values, or trajectory alignment are invalid.
        """
        fields = {
            "observations",
            "actions",
            "rewards",
            "terminations",
            "truncations",
            "infos",
            "artifacts",
        }
        if not isinstance(data, dict) or set(data) not in (fields, fields | {"error"}):
            raise ValueError("Malformed episode fields")
        if isinstance(data["rewards"], list) and data["rewards"][:1] == ["list"]:
            data = {name: _LegacyDecoder.decode(value) for name, value in data.items()}
        return cls.model_validate(data).validate_complete()

    def decode(self, data: dict) -> "Episode":
        """Replace this episode's fields only after the input validates.

        Args:
            data: Encoded episode accepted by ``from_data``.

        Returns:
            This episode with its fields replaced.

        Raises:
            ValueError: Input is invalid; the original episode remains unchanged.
        """
        self.__dict__.update(vars(type(self).from_data(data)))
        return self

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        """Sum of recorded rewards, including partial rewards for failed episodes."""
        return sum(self.rewards)

    @property
    def final_step(self) -> tuple:
        """Last ``(observation, reward, terminated, truncated, info)`` tuple.

        The reward belongs to the final transition, not the entire episode.

        Raises:
            IndexError: The episode has no recorded transition.
        """
        return (
            self.observations[-1],
            self.rewards[-1],
            self.terminations[-1],
            self.truncations[-1],
            self.infos[-1],
        )


class _LegacyDecoder:
    """Read existing tagged JSON episodes; new episodes use pickle."""

    MAX_ARRAY_BYTES = 262_144

    @staticmethod
    def _dtype(name):
        dtype = np.dtype(name)
        if dtype.kind not in "biufc" or dtype.itemsize > 16:
            raise ValueError("Only ordinary numeric and boolean arrays are supported")
        return dtype

    @classmethod
    def _shape(cls, shape, dtype):
        if not isinstance(shape, (list, tuple)) or len(shape) > 32:
            raise ValueError("Invalid array shape")
        if any(type(n) is not int or n < 0 or n > cls.MAX_ARRAY_BYTES for n in shape):
            raise ValueError("Invalid array dimension")
        size = math.prod(shape) * dtype.itemsize
        if size > cls.MAX_ARRAY_BYTES:
            raise ValueError("Array exceeds 256 KiB")
        return size

    @classmethod
    def decode(cls, value, _depth=0):
        if _depth > 64:
            raise ValueError("Value nesting exceeds 64")
        if value is None or type(value) in (bool, int, float, str):
            return value
        if not isinstance(value, list) or not value:
            raise ValueError("Invalid encoded value")
        tag = value[0]
        if tag == "array" and len(value) == 4:
            dtype = cls._dtype(value[1])
            size = cls._shape(value[2], dtype)
            if not isinstance(value[3], str) or len(value[3]) > 4 * ((size + 2) // 3):
                raise ValueError("Invalid array payload size")
            data = base64.b64decode(value[3], validate=True)
            if len(data) != size:
                raise ValueError("Array payload does not match shape")
            return np.frombuffer(data, dtype=dtype).reshape(value[2]).copy()
        if len(value) != 2:
            raise ValueError("Invalid encoded value")
        if tag == "bytes" and isinstance(value[1], str):
            return base64.b64decode(value[1], validate=True)
        if tag == "float" and value[1] in ("nan", "inf", "-inf"):
            return float(value[1])
        if tag in ("tuple", "list") and isinstance(value[1], list):
            items = [cls.decode(v, _depth + 1) for v in value[1]]
            return tuple(items) if tag == "tuple" else items
        if tag == "dict" and isinstance(value[1], list):
            return {cls.decode(k, _depth + 1): cls.decode(v, _depth + 1) for k, v in value[1]}
        raise ValueError("Unknown value tag")
