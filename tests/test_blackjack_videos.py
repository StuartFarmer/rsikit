"""The Docker launcher must read a consistent host-side SQLite snapshot."""

import json
import shutil
import sqlite3
import subprocess
import sys
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory


class BlackjackVideoScriptTests(unittest.TestCase):
    def test_snapshot_includes_wal_and_is_cleaned_up(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            run = root / "runs/current"
            run.mkdir(parents=True)
            script = root / "scripts/blackjack-videos"
            shutil.copy2(Path(__file__).resolve().parents[1] / "scripts/blackjack-videos", script)
            launcher = root / "scripts/run"
            launcher.write_text(
                f"#!{sys.executable}\n"
                "import json, sqlite3, sys\n"
                "path = sys.argv[sys.argv.index('--database') + 1]\n"
                "with sqlite3.connect(path) as db:\n"
                " print(json.dumps([path, db.execute('SELECT * FROM marker').fetchall()]))\n"
            )
            launcher.chmod(0o755)
            with closing(sqlite3.connect(run / "run.sqlite")) as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("CREATE TABLE marker (value)")
                db.execute("INSERT INTO marker VALUES (42)")
                db.commit()
                self.assertTrue((run / "run.sqlite-wal").exists())
                result = subprocess.run(
                    ["bash", str(script), "runs/current"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                snapshot, rows = json.loads(result.stdout)
                self.assertEqual(rows, [[42]])
                self.assertNotEqual(root / snapshot, run / "run.sqlite")
                self.assertFalse((root / snapshot).exists())
                self.assertEqual(db.execute("SELECT * FROM marker").fetchall(), [(42,)])
