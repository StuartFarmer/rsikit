"""Run with python -m unittest tests.test_ocean_native (requires a C compiler)."""

import pickle
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


class OceanNativeTests(unittest.TestCase):
    def test_external_cap_maximum_includes_initial_and_newly_spawned_tiles(self):
        from research.ocean.native import Batch

        with Batch(range(1000), max_steps=1) as batch:
            batch.step(np.zeros(1000, dtype=np.int64))
            reported = [result["max_tile"] for result in batch.results]
            observed = 2 ** batch.observations.max(axis=1).astype(np.int64)
            np.testing.assert_array_equal(reported, observed)

    def test_seed_order_external_cap_and_invalid_actions(self):
        from research.ocean.native import Batch

        def play(seeds):
            with Batch(seeds, max_steps=12) as batch:
                initial = batch.observations.copy()
                for invalid in ([0], [0, np.nan], [0, 4], [0, 0.5], [[0], [1]]):
                    with self.assertRaises(ValueError):
                        batch.step(invalid)
                    np.testing.assert_array_equal(batch.observations, initial)
                while len(batch.active):
                    batch.step(np.full(len(batch.active), 2))
                self.assertEqual(batch.observations.shape, (2, 16))
                self.assertEqual(batch.observations.dtype, np.float32)
                results = batch.results
                self.assertEqual([r["seed"] for r in results], seeds)
                self.assertTrue(all(r["ending"] == "external_cap" for r in results))
                self.assertTrue(all(r["steps"] == 12 for r in results))
                batch.step([])
                self.assertEqual(results, batch.results)
                return {r["seed"]: r for r in results}

        self.assertEqual(play([7, 42]), play([42, 7]))
        self.assertEqual(play([7, 42]), play([7, 42]))

    def test_native_timeout_keeps_completed_score(self):
        from research.ocean.native import Batch

        with Batch([0], max_steps=2000) as batch:
            rewards = []
            while len(batch.active):
                rewards.append(float(batch.step([0])[0]))
            result = batch.results[0]
            self.assertEqual(result["ending"], "native_timeout")
            self.assertEqual(result["steps"], 1000)
            self.assertAlmostEqual(result["return"], sum(rewards), places=2)
            # Upstream has already reset; score must come from its completed log.
            self.assertEqual(np.count_nonzero(batch.observations), 2)

    def test_native_adapter_captures_terminal_and_external_scores(self):
        from research.ocean.native import SOURCE

        # White-box C fixture uses no production-only test hooks. Removing log
        # capture, fresh lifetime state, or completed-slot guards breaks this.
        program = r"""
#include <assert.h>
#include "adapter.c"
int main(void) {
    uint32_t seed = 42;
    int64_t active = 0;
    float observation[16], reward, action = 2;
    double stats[5] = {0};
    Slot *s = ocean_create(1, &seed, observation);
    assert(s && s->game.lifetime_max_tile == 0);
    assert(s->game.scaffolding_ratio == 0 && s->terminal == 0);
    unsigned char blocked[16] = {1,2,1,2,2,1,2,1,1,2,1,2,2,1,2,1};
    memcpy(s->game.grid, blocked, 16);
    update_stats(&s->game);
    s->game.score = 123;
    ocean_step(s, 1, &active, &action, 2000, &reward, stats);
    assert(stats[0] == 123 && stats[1] == 4 && stats[3] == 1 && stats[4] == 1);
    assert(s->game.score == 0 && s->game.log.n == 1);
    assert(reward < -1);
    ocean_step(s, 1, &active, &action, 2000, &reward, stats);
    assert(stats[0] == 123 && s->game.log.n == 1 && s->game.tick == 0);
    ocean_close(s, 1);

    memset(stats, 0, sizeof(stats));
    s = ocean_create(1, &seed, observation);
    assert(s->game.lifetime_max_tile == 0);
    memset(s->game.grid, 0, 16);
    s->game.grid[0][0] = s->game.grid[0][1] = 1;
    update_stats(&s->game);
    ocean_step(s, 1, &active, &action, 1, &reward, stats);
    assert(stats[0] == 4 && stats[1] == 4 && stats[3] == 1 && stats[4] == 3);
    assert(s->game.log.n == 0 && s->game.score == 4);
    ocean_close(s, 1);

    memset(stats, 0, sizeof(stats));
    s = ocean_create(1, &seed, observation);
    memset(s->game.grid, 0, 16);
    s->game.grid[0][0] = 2;
    s->game.tick = 999;
    s->game.score = 456;
    update_stats(&s->game);
    ocean_step(s, 1, &active, &action, 2000, &reward, stats);
    assert(stats[0] == 456 && stats[1] == 4 && stats[3] == 1000 && stats[4] == 2);
    ocean_close(s, 1);
}
"""
        import os

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "check.c"
            source.write_text(program)
            executable = Path(directory) / "check"
            subprocess.run(
                [
                    *shlex.split(os.environ.get("CC", "cc")),
                    "-std=c11",
                    "-D_DEFAULT_SOURCE",
                    "-I",
                    str(SOURCE),
                    str(source),
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run([str(executable)], check=True)

    def test_gym_template_is_pickleable_and_returns_same_episode(self):
        from research.ocean.native import Batch, OceanEnv

        env = pickle.loads(pickle.dumps(OceanEnv(max_steps=3)))
        self.addCleanup(env.close)
        observation, _ = env.reset(seed=9)
        self.assertTrue(env.observation_space.contains(observation))
        with Batch([9], max_steps=3) as batch:
            np.testing.assert_array_equal(observation, batch.observations)
            for _ in range(3):
                observation, reward, terminated, truncated, info = env.step([2])
                self.assertEqual(reward, float(batch.step([2])[0]))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info, batch.results[0])
            observation, _ = env.reset(seed=9)
        with Batch([9], max_steps=3) as fresh:
            np.testing.assert_array_equal(observation, fresh.observations)


if __name__ == "__main__":
    unittest.main()
