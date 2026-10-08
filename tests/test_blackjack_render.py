"""Rendering must preserve the simulation and the dealer's hidden information."""

import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from examples.blackjack import choose_action
from rsikit.envs import BlackjackEnv


class BlackjackRenderTests(unittest.TestCase):
    def test_task_factory_records_blackjack_video(self):
        import gymnasium as gym
        from imageio_ffmpeg import count_frames_and_secs

        from rsikit.envs.tasks import make_environment

        with TemporaryDirectory() as temporary:
            env = make_environment(
                "Blackjack", render_mode="rgb_array", max_steps=3, shoes_per_episode=2
            )
            self.assertIn("2 shoes", env.instructions)
            with gym.wrappers.RecordVideo(
                env, str(Path(temporary) / "videos"), disable_logger=True
            ) as recording:
                obs, _ = recording.reset(seed=0)
                for _ in range(3):
                    obs, _, done, truncated, _ = recording.step(choose_action(obs))
                self.assertFalse(done)
                self.assertTrue(truncated)
            (video,) = Path(temporary).rglob("*.mp4")
            self.assertEqual(count_frames_and_secs(str(video))[0], 4)

    def test_all_seed_video_has_one_frame_per_action_at_30_fps(self):
        from imageio_ffmpeg import count_frames_and_secs
        from moviepy import VideoFileClip

        from examples.blackjack_videos import render_policy

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = 0
            total = 0
            for seed in (0, 1):
                env = BlackjackEnv(shoes_per_episode=1)
                obs, _ = env.reset(seed=seed)
                actions = []
                reward = 0
                while True:
                    action = choose_action(obs)
                    actions.append(action)
                    obs, earned, done, _, _ = env.step(action)
                    reward += earned
                    if done:
                        break
                (root / f"{seed}.json").write_text(
                    json.dumps({"actions": actions, "reward": reward})
                )
                frames += len(actions)
                total += reward
                env.close()
            result = render_policy(
                {
                    "video": str(root / "test.mp4"),
                    "shoes": 1,
                    "seeds": [0, 1],
                    "traces": str(root),
                    "policy_id": "baseline",
                    "name": "Baseline",
                }
            )
            self.assertEqual(result["frames"], frames)
            self.assertEqual(result["total_reward"], total)
            self.assertEqual([s["seed"] for s in result["segments"]], [0, 1])
            self.assertEqual(result["segments"][1]["first_frame"], result["segments"][0]["frames"])
            self.assertEqual(count_frames_and_secs(str(root / "test.mp4"))[0], frames)
            with VideoFileClip(str(root / "test.mp4")) as clip:
                self.assertEqual(clip.fps, 30)
                self.assertEqual(clip.size, [1280, 720])

    def test_replay_is_faithful_and_rendering_is_read_only(self):
        from rsikit.envs.blackjack_render import BlackjackRenderer

        plain = BlackjackEnv()
        rendered = BlackjackRenderer(BlackjackEnv(), policy_name="Baseline agent")
        obs, _ = plain.reset(seed=301)
        other, _ = rendered.reset(seed=301)
        np.testing.assert_array_equal(obs, other)
        self.assertEqual(plain._shoe, rendered.unwrapped._shoe)
        self.assertEqual(
            Counter(card.identity for card in rendered.unwrapped._shoe), dict.fromkeys(range(52), 6)
        )
        for card in rendered.unwrapped._shoe:
            self.assertEqual(card, min(card.identity % 13 + 1, 10))
        initial = rendered.render()
        self.assertEqual(initial.shape, (720, 1280, 3))
        self.assertEqual(initial.dtype, np.uint8)
        total = 0
        for _ in range(68):
            action = choose_action(obs)
            obs, reward, done, truncated, info = plain.step(action)
            other, other_reward, other_done, other_truncated, other_info = rendered.step(action)
            np.testing.assert_array_equal(obs, other)
            self.assertEqual(
                (reward, done, truncated, info),
                (other_reward, other_done, other_truncated, other_info),
            )
            total += reward
        self.assertEqual(rendered.points[-1], total)
        self.assertEqual(len(rendered.points), 69)
        self.assertEqual((total, plain._hands, plain._bets), (3, [[8, 3, 1], [8, 5]], [2, 1]))
        frame = rendered.render()
        np.testing.assert_array_equal(frame, rendered.render())
        self.assertEqual(
            plain.np_random.bit_generator.state, rendered.unwrapped.np_random.bit_generator.state
        )
        # A different unrevealed hole card must not change any pixel or dealer total.
        dealer = rendered.unwrapped._dealer
        hole = dealer[1]
        dealer[1] = rendered.unwrapped._shoe[0]
        np.testing.assert_array_equal(frame, rendered.render())
        dealer[1] = hole
        while True:
            action = choose_action(obs)
            obs, reward, done, truncated, info = plain.step(action)
            other, other_reward, other_done, other_truncated, other_info = rendered.step(action)
            np.testing.assert_array_equal(obs, other)
            self.assertEqual(
                (reward, done, truncated, info),
                (other_reward, other_done, other_truncated, other_info),
            )
            if info.get("shoe_shuffled"):
                break
        rendered.step(0)
        self.assertEqual(rendered.render().shape, (720, 1280, 3))
        for card in rendered.unwrapped._shoe:
            self.assertEqual(card, min(card.identity % 13 + 1, 10))
        rendered.reset(seed=301)
        self.assertEqual(rendered.points, [0.0])
        np.testing.assert_array_equal(initial, rendered.render())
        plain.reset(seed=301)
        plain.reset()
        rendered.reset()
        self.assertEqual(plain._shoe, rendered.unwrapped._shoe)
        for card in rendered.unwrapped._shoe:
            self.assertEqual(card, min(card.identity % 13 + 1, 10))
        rendered.close()


if __name__ == "__main__":
    unittest.main()
