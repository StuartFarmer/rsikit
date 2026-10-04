"""The public runner orders rounds and rejects incomplete evidence."""

import asyncio
import unittest
from types import SimpleNamespace

import rsikit
from tests.test_episode_storage import trajectory


class SearchTests(unittest.IsolatedAsyncioTestCase):
    def test_episode_panels_validate_before_update(self):
        from rsikit.optimization import validate_results
        from tests.test_episode_storage import trajectory

        self.assertEqual(validate_results({"p": {0: trajectory(3)}}, ["p"]), {0})
        for result in ({True: trajectory()}, {0: 3}, {0: trajectory(float("nan"))}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                validate_results({"p": result}, ["p"])

    async def test_round_order_and_best(self):
        events = []
        agent = SimpleNamespace(done=False, best=None)

        async def propose():
            events.append("propose")
            return [SimpleNamespace(id=str(len(events)))]

        async def evaluate(policies):
            events.append("evaluate")
            return {p.id: {0: trajectory(3)} for p in policies}

        def update(results):
            events.append("update")
            agent.best = next(iter(results))
            agent.done = len(events) == 6

        agent.propose, agent.update = propose, update
        self.assertEqual(await rsikit.search(agent, evaluate), "4")
        self.assertEqual(events, ["propose", "evaluate", "update"] * 2)
        self.assertEqual(await rsikit.search(agent, evaluate), "4")
        self.assertEqual(len(events), 6)

    async def test_invalid_rounds_never_update(self):
        policy = SimpleNamespace(id="p")
        for proposals, results in (
            ([policy], {}),
            ([policy], {"other": {0: trajectory(1)}}),
            ([policy], {"p": 1}),
            ([policy, policy], {}),
            ([], {}),
        ):

            async def propose():
                return proposals

            async def evaluate(policies):
                return results

            agent = SimpleNamespace(
                done=False,
                best=None,
                propose=propose,
                update=lambda _: self.fail("invalid results reached update"),
            )
            with (
                self.subTest(proposals=proposals, results=results),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                await rsikit.search(agent, evaluate)

    async def test_exception_checkpoints_preserve_primary_error(self):
        for error in (RuntimeError("offline"), asyncio.CancelledError()):
            checkpoints = []

            async def propose():
                raise error

            def checkpoint(agent):
                checkpoints.append(agent)
                raise ValueError("checkpoint failed")

            agent = SimpleNamespace(done=False, best=None, propose=propose)
            with self.assertLogs("rsikit.optimization", level="ERROR"):
                with self.assertRaises(type(error)) as raised:
                    await rsikit.search(agent, None, on_checkpoint=checkpoint)
            self.assertIs(raised.exception, error)
            self.assertEqual(checkpoints, [agent])
