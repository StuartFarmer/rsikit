"""Build the pinned Ocean Maze headless episode adapter (no renderer or map bank)."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

UPSTREAM = "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"
SOURCE_SHA256 = "c9992ce1536e9abb3b5a72240ce66d67554d017e3fdca979ed46edfa8ffa2900"


def headless(source):
    """Retain upstream generation, observations and movement at this exact source hash."""
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unexpected pinned Maze source hash")
    source = source.decode()

    def section(start, end):
        return source[source.index(start) : source.index(end)]

    code = """#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>
typedef unsigned char obs_t;
typedef struct Env Env;
typedef struct { obs_t* observations; float* actions; float* rewards; float* terminals; } Agent;
typedef void Renderer;
typedef Env Grid;
"""
    code += section("#define ATN_PASS", "typedef struct {\n    int cell_size;")
    code += section("typedef struct {\n    int width;", "void puf_reset")
    code += section("int move_to", "// Hold Left Shift")
    step = section("void puf_step", "Renderer* init_renderer")
    step = step.replace(
        "    maze_human_controls(env);",
        """
    if (env->agents[0].terminals[0]) {
        env->agents[0].rewards[0] = 0;
        return;
    }""",
    )
    step = step.replace(
        "if (env->tick >= 2*s->width*s->height)",
        "if (!env->agents[0].terminals[0] && env->tick >= 2*s->width*s->height)",
    )
    step = step[: step.rindex("    if (env->agents[0].terminals[0]) {")] + "}\n"
    code += step
    code += section("void generate_growing_tree_maze", "State* make_maze_levels")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Previously downloaded pinned maze.h")
    args = parser.parse_args()
    source = (
        args.source.read_bytes()
        if args.source
        else urlopen(
            f"https://raw.githubusercontent.com/PufferAI/PufferLib/{UPSTREAM}/ocean/maze/maze.h",
            timeout=30,
        ).read()
    )
    code = headless(source)
    directory = Path(sys.prefix) / "share" / "rsikit-maze"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "maze.h").write_bytes(source)
    adapter = Path(__file__).parent.parent / "research" / "ocean" / "maze.c"
    (directory / "headless.c").write_text(code + adapter.read_text())
    target = directory / "maze.so"
    temporary = directory / "maze.tmp.so"
    subprocess.run(
        [
            os.environ.get("CC", "cc"),
            "-O2",
            "-shared",
            "-fPIC",
            "-std=c11",
            "-D_DEFAULT_SOURCE",
            str(directory / "headless.c"),
            "-lm",
            "-o",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(target)
    print(f"Installed headless Ocean Maze from {UPSTREAM} at {target}")


if __name__ == "__main__":
    main()
