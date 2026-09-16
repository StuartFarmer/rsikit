"""Fixed example evaluator: interface/finite-output checks, then mean squared error.

This script checks and executes candidate arithmetic. Execute only trusted
candidates locally; use a sandboxed worker for arbitrary model output. The test
grid is public and is a demonstration objective, not a held-out benchmark.
"""

import argparse
import ast
import json
import math
from pathlib import Path


def evaluate(program: Path) -> dict:
    try:
        tree = ast.parse(program.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            raise ValueError("Define exactly one function: approximate(x)")
        function = functions[0]
        arguments = ast.parse("def approximate(x): return x").body[0].args
        if (
            function.name != "approximate"
            or ast.dump(function.args) != ast.dump(arguments)
            or function.decorator_list
            or function.returns is not None
            or getattr(function, "type_params", [])
            or len(function.body) != 1
            or not isinstance(function.body[0], ast.Return)
        ):
            raise ValueError("approximate(x) must contain one arithmetic return expression")
        for node in tree.body:
            if node is not function and not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                raise ValueError("Only a module docstring and approximate(x) are allowed")
        allowed = (
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Name,
            ast.Load,
            ast.Constant,
        )
        for node in ast.walk(function.body[0].value):
            if (
                not isinstance(node, allowed)
                or isinstance(node, ast.Name)
                and node.id != "x"
                or isinstance(node, ast.Constant)
                and type(node.value) not in (int, float)
            ):
                raise ValueError("Use only x, numbers and arithmetic; no calls or imports")
        # Compile the actual bytes every time; timestamp-based import caches can be stale.
        namespace = {}
        exec(compile(tree, str(program), "exec"), namespace)
        errors = []
        for index in range(201):
            x = -1 + index / 100
            value = float(namespace["approximate"](x))
            if not math.isfinite(value):
                raise ValueError("approximate must return finite values")
            errors.append((value - math.sin(x)) ** 2)
        mse = math.fsum(errors) / len(errors)
        if not math.isfinite(mse):
            raise ValueError("non-finite mean squared error")
        return {
            "valid": True,
            "metrics": {"mse": mse},
            "feedback": "201 finite outputs on [-1, 1].",
        }
    except Exception as exc:
        # Candidate validation/execution is the boundary; infrastructure stays outside it.
        return {"valid": False, "metrics": {}, "feedback": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.program)
    args.output.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, allow_nan=False))
