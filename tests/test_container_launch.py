"""The host launcher preserves arguments and delegates lifecycle to Docker."""

import io
import json
import os
import pty
import shlex
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def test_arguments_credentials_paths_and_exit_status(self):
        with tempfile.TemporaryDirectory(prefix="rsikit launch ") as directory:
            root = Path(directory)
            docker = root / "docker"
            docker.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ["CALLS"], "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "image": print("sha256:test")
sys.exit(int(os.environ.get("BUILD_EXIT", "0")) if sys.argv[1] == "build" else
         int(os.environ.get("RUN_EXIT", "0")) if sys.argv[1] == "run" else 0)
""")
            docker.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
            env.update(
                PATH=f"{root}:{env['PATH']}",
                CALLS=str(root / "calls"),
                RSIKIT_RUNS_DIR=str(root / "output with spaces"),
                RUN_EXIT="17",
                ANTHROPIC_API_KEY="sentinel",
                RSIKIT_ENV_FILE=str(root / "keys.env"),
            )
            Path(env["RSIKIT_ENV_FILE"]).write_text("OPENAI_API_KEY=file-sentinel\n")
            result = subprocess.run(
                [str(ROOT / "scripts/run"), "examples.elitesearch", "--name", "a name with spaces"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 17, result.stderr)
            calls = [json.loads(line) for line in (root / "calls").read_text().splitlines()]
            self.assertEqual(calls[0], ["build", "-t", "rsikit:local", str(ROOT)])
            run = calls[-1]
            self.assertEqual(
                run[-7:],
                [
                    "rsikit:local",
                    "python",
                    "-u",
                    "-m",
                    "examples.elitesearch",
                    "--name",
                    "a name with spaces",
                ],
            )
            self.assertIn(f"{root / 'output with spaces'}:/app/runs", run)
            self.assertIn("ANTHROPIC_API_KEY", run)
            self.assertNotIn("OPENAI_API_KEY", run)
            self.assertNotIn("sentinel", " ".join(run))
            self.assertNotIn("-t", run)
            self.assertIn("--read-only", run)
            self.assertIn("RSIKIT_IMAGE_ID=sha256:test", run)
            master, slave = pty.openpty()
            try:
                terminal = subprocess.run(
                    [str(ROOT / "scripts/run"), "tests"],
                    env=env,
                    stdin=slave,
                    stdout=slave,
                    stderr=slave,
                    timeout=5,
                )
                self.assertEqual(terminal.returncode, 17)
                terminal_run = json.loads((root / "calls").read_text().splitlines()[-1])
                self.assertIn("-it", terminal_run)
                self.assertIn("TERM=xterm-256color", terminal_run)
                self.assertIn("COLORTERM", terminal_run)
            finally:
                os.close(master)
                os.close(slave)
            env["BUILD_EXIT"] = "19"
            self.assertEqual(
                subprocess.run(
                    [str(ROOT / "scripts/run"), "tests"], env=env, capture_output=True
                ).returncode,
                19,
            )

    def test_notebook_commands_use_launcher_and_mounted_paths(self):
        notebook = json.loads((ROOT / "examples/elitelist_papers/paper1.ipynb").read_text())
        scope = dict(
            Path=Path,
            ROOT=ROOT,
            WORK=ROOT / "examples/elitelist_papers",
            RUNS=ROOT / "runs",
            LAUNCHER=ROOT / "scripts/run",
            shlex=shlex,
        )
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell {i}", "exec")
        with redirect_stdout(io.StringIO()):
            exec("".join(notebook["cells"][4]["source"]), scope)
            self.assertEqual(len(scope["commands"]), 60)
            for name, command in scope["commands"]:
                self.assertEqual(
                    command[:2], [str(ROOT / "scripts/run"), "examples.elitelist_papers.run"]
                )
                self.assertEqual(command[command.index("--output") + 1], f"runs/{name}")
            resume = "".join(notebook["cells"][6]["source"]).replace(
                "RESUME_RUN = None", "RESUME_RUN = RUNS / 'saved run'"
            )
            exec(resume, scope)
            self.assertEqual(scope["command"][-2:], ["--resume", "runs/saved run"])

    def test_missing_docker(self):
        env = {**os.environ, "PATH": "/usr/bin:/bin"}
        result = subprocess.run(
            [str(ROOT / "scripts/run"), "tests"], env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 127)
        self.assertIn("Docker is required", result.stderr)

    def test_usage_and_poker_are_rejected_before_build(self):
        for arguments in ([], ["examples.elitelist_papers.poker"]):
            result = subprocess.run([str(ROOT / "scripts/run"), *arguments], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(result.stderr)


if __name__ == "__main__":
    unittest.main()
