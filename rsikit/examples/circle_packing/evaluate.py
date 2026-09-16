"""Validate and score returned circle data on the host; never execute source here."""

import argparse
import ast
import json
import math
from pathlib import Path

COUNT = 10
TOLERANCE = 1e-9


def read_circles(source: str) -> list:
    """Read legacy literal-only runs for plotting, without executing their source."""
    if len(source) > 65_536:
        raise ValueError("Source exceeds 64 KiB")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Define only pack_circles(); no imports or other statements")
    function = tree.body[0]
    if (
        function.name != "pack_circles"
        or ast.dump(function.args) != ast.dump(ast.parse("def pack_circles(): pass").body[0].args)
        or function.decorator_list
        or function.returns is not None
        or getattr(function, "type_params", [])
        or len(function.body) != 1
        or not isinstance(function.body[0], ast.Return)
    ):
        raise ValueError("pack_circles() must contain exactly one literal return statement")
    return validate_circles(ast.literal_eval(function.body[0].value))


def validate_circles(circles: list) -> list:
    if not isinstance(circles, list) or len(circles) != COUNT:
        raise ValueError(f"Return a list of exactly {COUNT} (x, y, radius) triples")
    for index, circle in enumerate(circles):
        if not isinstance(circle, (list, tuple)) or len(circle) != 3:
            raise ValueError(f"Circle {index}: expected (x, y, radius)")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in circle):
            raise ValueError(f"Circle {index}: coordinates/radius must be finite numbers")
        if not 0 < circle[2] <= 0.5:
            raise ValueError(f"Circle {index}: radius must be positive and at most 0.5")
    return circles


def evaluate(circles: list) -> dict:
    try:
        circles = validate_circles(circles)
        violations, margins = [], []
        for i, (x, y, radius) in enumerate(circles):
            margin = min(x - radius, y - radius, 1 - x - radius, 1 - y - radius)
            margins.append(margin)
            if margin < -TOLERANCE:
                violations.append(f"Circle {i} crosses the square boundary by {-margin:.9g}")
            for j in range(i):
                other_x, other_y, other_radius = circles[j]
                gap = math.hypot(x - other_x, y - other_y) - radius - other_radius
                margins.append(gap)
                if gap < -TOLERANCE:
                    violations.append(f"Circles {j} and {i} overlap by {-gap:.9g}")
        if violations:
            return {"valid": False, "metrics": {}, "feedback": "; ".join(violations)}
        return {
            "valid": True,
            "metrics": {"sum_radii": math.fsum(c[2] for c in circles), "min_margin": min(margins)},
            "feedback": f"All {COUNT} circles fit and do not overlap (tolerance {TOLERANCE:g}).",
        }
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        return {"valid": False, "metrics": {}, "feedback": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.circles.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, allow_nan=False))
