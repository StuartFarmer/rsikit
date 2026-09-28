"""Historical variants consume episode evidence through the common optimizer API."""

import unittest
from pathlib import Path
from unittest.mock import patch

from slick import prompts

from research import alphaevolve
from research.alphaevolve import improved, original
from rsikit import Optimizer
from tests.providers import ScriptedProvider
from tests.test_alphaevolve import program
from tests.test_episode_storage import trajectory


class OptimizerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_baselines_propose_and_update_from_episode_returns(self):
        with patch.object(prompts, "TEMPLATE_ROOT", Path(alphaevolve.__file__).parent):
            for variant in (original, improved):
                with self.subTest(variant=variant.__name__):
                    agent: Optimizer = variant.AlphaEvolve(
                        "task", ScriptedProvider([program(0)]), config=variant.Config(islands=1)
                    )
                    (policy,) = await agent.propose(1)
                    agent.update([(policy, trajectory(3)), (policy, trajectory(7))])
                    self.assertEqual(agent.best.id, policy.id)
                    self.assertEqual(agent.islands[0].score, 5)
                    self.assertEqual(agent.completed, 1)
                    self.assertEqual(agent._pending, {})
