"""Run one episode on private inputs, capturing logs and artifacts."""

import asyncio
import logging
import os
import pickle
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import cloudpickle
import gymnasium as gym
from huey.exceptions import TaskTimeout

from ..episode import Episode
from ..evaluation import Evaluator, InfrastructureError, PolicyError, _policy_boundary
from ..policy import Policy, load_policy

MAX_RESULT = 64 * 1024 * 1024


def run_episode(definition, seed, max_steps, instructions):
    """Evaluate one private input pair, restoring the reused worker's cwd and output."""
    sys.stdout.flush()
    sys.stderr.flush()
    with (
        TemporaryDirectory(prefix="rsikit-episode-") as directory,
        open(Path(directory) / "episode.log", "w", buffering=1) as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        cwd = os.getcwd()
        stdout, stderr = os.dup(1), os.dup(2)
        os.chdir(directory)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)

        async def evaluate():
            policy = candidate if isinstance(candidate, Policy) else None
            episode = Episode()
            try:
                if policy is None:
                    text = instructions
                    if text is None:
                        text = (
                            env.get_wrapper_attr("instructions")
                            if env.has_wrapper_attr("instructions")
                            else ""
                        )
                    with _policy_boundary():
                        policy = load_policy(
                            candidate.source,
                            deepcopy(env.observation_space),
                            deepcopy(env.action_space),
                            text,
                        )
                episode = await Evaluator(max_steps=max_steps).evaluate(policy, env, seed=seed)
                return episode
            except PolicyError as exc:
                episode.error = "".join(traceback.format_exception(exc))
                return episode
            finally:
                primary = sys.exc_info()[1]
                try:
                    try:
                        if policy is not None:
                            try:
                                with _policy_boundary():
                                    await policy.close()
                            except PolicyError as exc:
                                if primary is None and episode.error is None:
                                    episode.error = "".join(traceback.format_exception(exc))
                                else:
                                    logging.getLogger(__name__).exception(
                                        "Cleanup failed while handling an episode error"
                                    )
                    finally:
                        env.close()
                except BaseException:
                    if primary is None and episode.error is None:
                        raise
                    # Keep the original failure/cancellation; expose secondary cleanup errors.
                    logging.getLogger(__name__).exception(
                        "Cleanup failed while handling an episode error"
                    )

        try:
            candidate, env = cloudpickle.loads(definition)
            wrapper, index = env, 0
            while isinstance(wrapper, gym.Wrapper):
                if isinstance(wrapper, gym.wrappers.RecordVideo):
                    wrapper.video_folder = str(Path(directory) / "videos" / str(index))
                    Path(wrapper.video_folder).mkdir(parents=True)
                    index += 1
                wrapper = wrapper.env
            try:
                episode = asyncio.run(evaluate())
            except TaskTimeout as exc:
                episode = Episode(error=str(exc))
            sys.stdout.flush()
            sys.stderr.flush()
            if episode.infos:
                episode.artifacts.update(episode.infos[-1].get("artifacts", {}))
            size = sum(len(data) for data in episode.artifacts.values())
            for path in Path(directory).rglob("*"):
                if path.is_file() and not path.is_symlink():
                    size += path.stat().st_size
                    if size > MAX_RESULT:
                        raise InfrastructureError("Evaluation artifacts exceed 64 MiB")
                    if path.stat().st_size:
                        episode.artifacts[str(path.relative_to(directory))] = path.read_bytes()
            data = episode.encode()
            if len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)) > MAX_RESULT:
                raise InfrastructureError("Episode exceeds 64 MiB")
            return data
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout, 1)
            os.dup2(stderr, 2)
            os.close(stdout)
            os.close(stderr)
            os.chdir(cwd)
