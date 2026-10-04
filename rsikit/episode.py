"""Episode trajectories returned by Evaluator."""

import base64
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
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
    rewards: list[float] = field(default_factory=list)
    terminations: list[bool] = field(default_factory=list)
    truncations: list[bool] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, bytes] = field(default_factory=dict)
    error: str | None = None

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


MAX_ARRAY_BYTES = 262_144


def _dtype(name):
    dtype = np.dtype(name)
    if dtype.kind not in "biufc" or dtype.itemsize > 16:
        raise ValueError("Only ordinary numeric and boolean arrays are supported")
    return dtype


def _shape(shape, dtype):
    if not isinstance(shape, (list, tuple)) or len(shape) > 32:
        raise ValueError("Invalid array shape")
    if any(type(n) is not int or n < 0 or n > MAX_ARRAY_BYTES for n in shape):
        raise ValueError("Invalid array dimension")
    size = math.prod(shape) * dtype.itemsize
    if size > MAX_ARRAY_BYTES:
        raise ValueError("Array exceeds 256 KiB")
    return size


def encode(value, _depth=0):
    if _depth > 64:
        raise ValueError("Value nesting exceeds 64")
    if isinstance(value, np.ndarray):
        _shape(value.shape, _dtype(value.dtype.str))
        return [
            "array",
            value.dtype.str,
            list(value.shape),
            base64.b64encode(value.tobytes()).decode(),
        ]
    if isinstance(value, np.generic):
        return encode(value.item(), _depth + 1)
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode()]
    if type(value) is float:
        return value if math.isfinite(value) else ["float", str(value)]
    if isinstance(value, (tuple, list)):
        return [
            "tuple" if isinstance(value, tuple) else "list",
            [encode(v, _depth + 1) for v in value],
        ]
    if isinstance(value, dict):
        return ["dict", [[encode(k, _depth + 1), encode(v, _depth + 1)] for k, v in value.items()]]
    raise ValueError(f"Unsupported value type: {type(value).__name__}")


def decode(value, _depth=0):
    if _depth > 64:
        raise ValueError("Value nesting exceeds 64")
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError("Invalid encoded value")
    tag = value[0]
    if tag == "array" and len(value) == 4:
        dtype = _dtype(value[1])
        size = _shape(value[2], dtype)
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
        items = [decode(v, _depth + 1) for v in value[1]]
        return tuple(items) if tag == "tuple" else items
    if tag == "dict" and isinstance(value[1], list):
        return {decode(k, _depth + 1): decode(v, _depth + 1) for k, v in value[1]}
    raise ValueError("Unknown value tag")


def encode_episode(episode):
    """Stable data-only representation for saved episodes."""
    if not isinstance(episode, Episode):
        raise ValueError("Expected an Episode")
    _validate_episode(vars(episode))
    return {
        name: encode(value)
        for name, value in vars(episode).items()
        if name != "error" or value is not None
    }


def decode_episode(data):
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
    values = {name: decode(value) for name, value in data.items()}
    values.setdefault("error", None)
    _validate_episode(values)
    return Episode(**values)


def _validate_episode(values):
    if any(not isinstance(values[name], list) for name in set(values) - {"artifacts", "error"}):
        raise ValueError("Episode tracks must be lists")
    error = values.get("error")
    if error is not None and (not isinstance(error, str) or not error.strip()):
        raise ValueError("Episode error must be nonempty text")
    length = len(values["rewards"])
    if (not length and error is None) or any(
        len(values[name]) != length for name in ("actions", "terminations", "truncations")
    ):
        raise ValueError("Episode transitions are not aligned")
    initial = len(values["observations"])
    if initial != len(values["infos"]) or (
        initial != length + 1 and not (error is not None and length == initial == 0)
    ):
        raise ValueError("Episode must include its initial observation and info")
    if any(type(r) not in (int, float) or not math.isfinite(r) for r in values["rewards"]):
        raise ValueError("Episode rewards must be finite numbers")
    ends = list(zip(values["terminations"], values["truncations"]))
    if any(type(flag) is not bool for pair in ends for flag in pair):
        raise ValueError("Episode end flags must be boolean")
    if any(a or b for a, b in ends[:-1]) or (error is None and not any(ends[-1])):
        raise ValueError("Episode must end exactly at its last transition")
    if any(not isinstance(info, dict) for info in values["infos"]):
        raise ValueError("Episode infos must be dictionaries")
    artifacts = values["artifacts"]
    if not isinstance(artifacts, dict) or any(
        not isinstance(k, str) or not isinstance(v, bytes) for k, v in artifacts.items()
    ):
        raise ValueError("Episode artifacts must map paths to bytes")
