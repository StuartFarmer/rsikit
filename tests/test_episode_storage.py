"""Raw episode transport and persistence do not require an execution backend."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rsikit import Episode, PolicyDefinition, Run
from rsikit.episode import EpisodeEncoder


def trajectory(reward=3.0, artifacts=None):
    return Episode(
        observations=[np.array([0.0]), np.array([1.0])],
        actions=[1],
        rewards=[float(reward)],
        terminations=[True],
        truncations=[False],
        infos=[{}, {"detail": (1, 2)}],
        artifacts=artifacts or {},
    )


class EpisodeStorageTests(unittest.TestCase):
    def test_constructor_validates_fields_while_allowing_incomplete_rollouts(self):
        self.assertEqual(len(Episode()), 0)
        self.assertEqual(Episode(observations=[0], infos=[{}]).observations, [0])
        for values in (
            {"observations": ()},
            {"rewards": [True]},
            {"rewards": [np.bool_(True)]},
            {"rewards": ["1"]},
            {"rewards": [float("nan")]},
            {"terminations": [1]},
            {"infos": [None]},
            {"artifacts": {"log": "text"}},
            {"error": " \n"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Episode(**values)

    def test_mutated_fields_are_revalidated_at_storage_and_feedback_boundaries(self):
        from rsikit.optimization import validate_results

        for name, value in (
            ("rewards", float("nan")),
            ("rewards", np.bool_(True)),
            ("terminations", 1),
            ("infos", None),
        ):
            episode = trajectory()
            getattr(episode, name)[0] = value
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    episode.encode()
                with self.assertRaises(ValueError):
                    validate_results({"p": {0: episode}}, ["p"])
        episode = trajectory()
        episode.artifacts["log"] = "text"
        with self.assertRaises(ValueError):
            episode.encode()

    def test_partial_failures_and_arbitrary_info_values_roundtrip(self):
        for episode in (
            Episode(error="initialization failed"),
            Episode(observations=[0], infos=[{}], error="reset failed"),
            Episode(
                observations=[0, 1],
                actions=[0],
                rewards=[1],
                terminations=[False],
                truncations=[False],
                infos=[{}, {2: (b"bytes", float("inf"))}],
                error="act failed",
            ),
        ):
            with self.subTest(error=episode.error):
                restored = Episode.from_data(episode.encode())
                self.assertEqual(restored, episode)
        with self.assertRaises(ValueError):
            Episode().encode()

    def test_pre_migration_saved_json(self):
        # Literal old-format fixture, independent of the current encoder.
        saved = {
            "observations": ["list", [0, 1]],
            "actions": ["list", [0]],
            "rewards": ["list", [7.0]],
            "terminations": ["list", [True]],
            "truncations": ["list", [False]],
            "infos": ["list", [["dict", []], ["dict", []]]],
            "artifacts": ["dict", [["log.txt", ["bytes", "aGk="]]]],
        }
        episode = Episode.from_data(saved)
        self.assertEqual(episode.total_reward, 7.0)
        self.assertEqual(episode.artifacts, {"log.txt": b"hi"})
        self.assertEqual(episode.encode(), saved)

    def test_decode_replaces_existing_episode_only_after_validation(self):
        episode = Episode(error="previous failure")
        saved = trajectory(artifacts={"log.txt": b"hello"}).encode()
        self.assertIs(episode.decode(saved), episode)
        self.assertEqual(episode.encode(), saved)
        self.assertIsNone(episode.error)

        broken = dict(saved, actions=["list", []])
        with self.assertRaises(ValueError):
            episode.decode(broken)
        self.assertEqual(episode.encode(), saved)

    def test_codec_roundtrip_preserves_values_and_rejects_misalignment(self):
        episode = trajectory(artifacts={"log.txt": b"hello"})
        data = episode.encode()
        restored = Episode.from_data(data)
        np.testing.assert_array_equal(restored.observations[0], [0.0])
        self.assertEqual(restored.actions, [1])
        self.assertEqual(restored.rewards, [3.0])
        self.assertEqual(restored.infos, [{}, {"detail": (1, 2)}])
        self.assertEqual(restored.artifacts, {"log.txt": b"hello"})
        for field, value in (
            ("actions", []),
            ("rewards", [float("nan")]),
            ("terminations", [False]),
            ("infos", [{}, None]),
        ):
            with self.subTest(field=field):
                broken = dict(data)
                broken[field] = EpisodeEncoder.encode(value)
                with self.assertRaises(ValueError):
                    Episode.from_data(broken)

    def test_run_reopens_without_environment_and_keeps_checkpoint_and_episode(self):
        policy = PolicyDefinition.from_text(
            "from rsikit import Policy\nclass Solution(Policy):\n    async def act(self, observation):\n        return 0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            with Run.create(name="optimizer", path=path) as run:
                run.save_policy(policy, scores={42: 9.0})
                run.save_episode(policy, 42, trajectory(artifacts={"nested/log.txt": b"ok"}))
                self.assertEqual(run.scores(policy), {42: 9.0})  # Explicit optimizer score.
            with Run.open(path) as run:
                restored = run.load_episode(policy, 42)
                self.assertEqual(restored.total_reward, 3.0)
                self.assertEqual(restored.artifacts, {"nested/log.txt": b"ok"})
                self.assertEqual(run.policies()[0].id, policy.id)
                self.assertFalse(hasattr(run, "evaluate"))
                self.assertFalse(hasattr(run, "executor"))
                with self.assertRaises(ValueError):
                    run.save_episode(policy, 43, trajectory(artifacts={"../../escape": b"no"}))


if __name__ == "__main__":
    unittest.main()
