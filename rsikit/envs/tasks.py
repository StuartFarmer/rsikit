"""Shared Gymnasium task presets used by search examples and replay."""

import gymnasium as gym

from . import BitcoinEnv, BlackjackEnv

TASKS = {
    "Bitcoin": "Maximize final USD wealth after BTC trading fees over the training period.",
    "Blackjack": "Maximize net profit over a finite blackjack shoe, with betting and sitting out.",
    "CartPole-v1": (
        "Balance the pole for as many steps as possible. Observation: [cart position, "
        "cart velocity, pole angle, pole angular velocity]. Action 0 pushes left; "
        "1 pushes right. Each surviving step earns 1 reward. The episode terminates "
        "when the pole tilts too far or the cart leaves the allowed range."
    ),
    "LunarLander-v3": (
        "Land safely near the pad at (0, 0), maximizing cumulative reward. Wind and "
        "turbulence are enabled. Observation indices: 0 horizontal position, 1 vertical "
        "position, 2 horizontal velocity, 3 vertical velocity, 4 angle, 5 angular "
        "velocity, 6 left leg contact, 7 right leg contact. Positions and velocities "
        "are normalized simulator values. Return a float32 array [main, lateral], "
        "each in [-1, 1]. Main < 0 turns the main engine off; main in [0, 1] maps to "
        "50-100% power. Lateral in (-0.5, 0.5) turns side engines off; below -0.5 "
        "fires the left booster, above 0.5 fires the right, with magnitude mapping "
        "to 50-100% power. Reward favors approaching the pad, slowing down, staying "
        "upright and leg contact; firing engines costs reward. A crash costs 100; "
        "a safe landing earns 100. Termination: crash, leaving the horizontal bounds, "
        "or coming to rest. Initial force and wind vary with the episode seed."
    ),
    "BipedalWalker-v3": (
        "Walk to the right over uneven terrain without falling, maximizing cumulative "
        "reward. Normal terrain, not hardcore. Observation indices: 0 hull angle, "
        "1 scaled hull angular velocity, 2-3 scaled horizontal/vertical velocity; "
        "4 hip angle, 5 scaled hip speed, 6 knee angle plus 1, 7 scaled knee speed, "
        "8 foot contact for the first leg; 9-13 the same five values for the second "
        "leg; 14-23 ten lidar fractions (0 near, 1 far). Return a float32 array of "
        "four motor commands in [-1, 1]: first hip, first knee, second hip, second "
        "knee. Sign sets motor direction; magnitude limits torque. Reward favors "
        "forward progress and an upright hull, with a motor-effort penalty. Falling "
        "costs 100. Termination: hull contacts ground or reaching the terrain end. "
        "Terrain and initial push vary with the episode seed."
    ),
}


def make_environment(name, *, max_steps=None, render_mode=None, shoes_per_episode=24):
    if name in ("Bitcoin", "Blackjack"):
        if render_mode is not None:
            raise ValueError(f"{name} does not support rendering")
        env = (
            BitcoinEnv() if name == "Bitcoin" else BlackjackEnv(shoes_per_episode=shoes_per_episode)
        )
        if max_steps is not None:
            instructions = env.instructions
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
            env.instructions = instructions + f" The episode is truncated after {max_steps} steps."
        return env
    options = {"continuous": True, "enable_wind": True} if name == "LunarLander-v3" else {}
    if max_steps is not None:
        options["max_episode_steps"] = max_steps
    # Keep instructions on a wrapper: Box2D's EzPickle reconstructs the base env.
    env = gym.Wrapper(gym.make(name, render_mode=render_mode, **options))
    env.instructions = (
        f"{name}: {TASKS[name]}\n"
        f"Observation space: {env.observation_space}\nAction space: {env.action_space}\n"
        f"The episode is truncated after {env.spec.max_episode_steps} steps."
    )
    return env
