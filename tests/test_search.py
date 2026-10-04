"""The public runner orders rounds and rejects incomplete evidence."""

import asyncio
import unittest
from types import SimpleNamespace

import rsikit


class SearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertTrue(hasattr(rsikit, "Measurement"), "core owns Measurement")
        self.assertTrue(hasattr(rsikit, "search"), "core owns search")

    def test_measurement_preserves_evidence_and_rejects_invalid_values(self):
        from research.rewards import Measurement

        self.assertIs(Measurement, rsikit.Measurement)
        result = Measurement({0: 0, 1: 10}, "ok", metrics={"cost": -3}, features={"size": 2})
        self.assertEqual(result.scores, {0: 0, 1: 10})
        self.assertEqual(result.features, {"size": 2})
        self.assertFalse(Measurement(failure="broken").accepted)
        for values in (
            {},
            {"scores": {0: True}},
            {"scores": {"0": 1}},
            {"metrics": {"": 1}},
            {"metrics": {"x": float("nan")}},
            {"features": {"x": float("inf")}},
            {"scores": {0: 1}, "feedback": 3},
            {"scores": {0: 1}, "accepted": 1},
            {"failure": ""},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Measurement(**values)

    async def test_round_order_and_best(self):
        events = []
        agent = SimpleNamespace(done=False, best=None)

        async def propose():
            events.append("propose")
            return [SimpleNamespace(id=str(len(events)))]

        async def evaluate(policies):
            events.append("evaluate")
            return {p.id: rsikit.Measurement({0: 3}) for p in policies}

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
            ([policy], {"other": rsikit.Measurement({0: 1})}),
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
