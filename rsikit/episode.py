"""Episode trajectories returned by Evaluator."""

import base64
import math
from dataclasses import field
from typing import Annotated, Any

import numpy as np
from pydantic import BeforeValidator, ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass


def _reward(value):
    # Strict float validation also accepts float-convertible objects, including NumPy booleans.
    if type(value) not in (int, float):
        raise ValueError("Episode rewards must be ordinary numbers")
    return value


@dataclass(
    config=ConfigDict(
        strict=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )
)
class Episode:
    """An attempted rollout, including its partial trajectory and candidate error.

    Recorded transitions have T actions/rewards/flags and T+1 observations/infos.
    Initialization failures may have no initial observation. An error means the
    accumulated rewards are partial evidence, not a successful fitness score.

    Transition t is observations[t], actions[t], rewards[t], observations[t+1],
    terminations[t], truncations[t], infos[t+1]. Index 0 of infos is reset info.
    """

    observations: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    rewards: list[Annotated[int | float, BeforeValidator(_reward)]] = field(default_factory=list)
    terminations: list[bool] = field(default_factory=list)
    truncations: list[bool] = field(default_factory=list)
    infos: list[dict[Any, Any]] = field(default_factory=list)
    artifacts: dict[str, bytes] = field(default_factory=dict)
    error: Annotated[str, Field(pattern=r"\S")] | None = None

    def validate_complete(self) -> "Episode":
        """Validate current fields and trajectory, including post-construction mutations."""
        _EPISODE.validate_python(self)
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
        """Return the stable data-only representation for a saved episode."""
        self.validate_complete()
        return {
            name: EpisodeEncoder.encode(value)
            for name, value in vars(self).items()
            if name != "error" or value is not None
        }

    @classmethod
    def from_data(cls, data: dict) -> "Episode":
        """Construct and validate an episode from its encoded representation."""
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
        values = {name: EpisodeEncoder.decode(value) for name, value in data.items()}
        return cls(**values).validate_complete()

    def decode(self, data: dict) -> "Episode":
        """Replace this episode with decoded data, leaving it intact if validation fails."""
        self.__dict__.update(vars(type(self).from_data(data)))
        return self

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def final_step(self) -> tuple:
        """The last Gymnasium step tuple, including its per-step reward."""
        return (
            self.observations[-1],
            self.rewards[-1],
            self.terminations[-1],
            self.truncations[-1],
            self.infos[-1],
        )


_EPISODE = TypeAdapter(Episode)


class EpisodeEncoder:
    """Encode and decode the nested values stored in episode fields."""

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
    def encode(cls, value, _depth=0):
        if _depth > 64:
            raise ValueError("Value nesting exceeds 64")
        if isinstance(value, np.ndarray):
            cls._shape(value.shape, cls._dtype(value.dtype.str))
            return [
                "array",
                value.dtype.str,
                list(value.shape),
                base64.b64encode(value.tobytes()).decode(),
            ]
        if isinstance(value, np.generic):
            return cls.encode(value.item(), _depth + 1)
        if value is None or type(value) in (bool, int, str):
            return value
        if isinstance(value, bytes):
            return ["bytes", base64.b64encode(value).decode()]
        if type(value) is float:
            return value if math.isfinite(value) else ["float", str(value)]
        if isinstance(value, (tuple, list)):
            return [
                "tuple" if isinstance(value, tuple) else "list",
                [cls.encode(v, _depth + 1) for v in value],
            ]
        if isinstance(value, dict):
            return [
                "dict",
                [[cls.encode(k, _depth + 1), cls.encode(v, _depth + 1)] for k, v in value.items()],
            ]
        raise ValueError(f"Unsupported value type: {type(value).__name__}")

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
