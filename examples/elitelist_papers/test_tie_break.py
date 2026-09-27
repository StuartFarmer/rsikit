"""Score ties prefer this generation's edits, remixes, incumbents, then new ideas."""

import json
import unittest

from research.elitesearch import Config, EliteSearch, Generation, Organism
from rsikit.policy import _policy_class
from tests.providers import ScriptedProvider
from tests.test_elitesearch import program


class TieBreakTests(unittest.TestCase):
    def agent(self):
        return EliteSearch(
            "test",
            ScriptedProvider([]),
            None,
            config=Config(elite_size=2, population_size=5, generations=3),
            seed=42,
        )

    def test_score_then_operator_then_id(self):
        agent = self.agent()
        agent.config = Config(elite_size=10)
        agent.elites = [Organism(id=1, generation=1, kind="edit", score=7)]
        rows = [
            Organism(id=2, generation=2, kind="new", score=7),
            Organism(id=3, generation=2, kind="remix", score=7),
            Organism(id=4, generation=2, kind="edit", score=7),
            Organism(id=5, generation=2, kind="edit", score=7),
            Organism(id=6, generation=2, kind="new", score=8),
            Organism(id=7, generation=2, kind="edit", score=6),
            Organism(id=8, generation=2, kind="edit", score=None),
        ]
        generation = Generation(number=2)
        agent._promote(generation, list(reversed(rows)))
        self.assertEqual(generation.elite_ids, [6, 4, 5, 3, 1, 2, 7])
        self.assertEqual(generation.promoted_ids, [6, 4, 5, 3, 2, 7])

    def test_restore_preserves_old_and_new_boards_then_uses_new_rule(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                original = self.agent()
                for number in (1, 2):
                    generation = Generation(number=number)
                    original.generations.append(generation)
                    rows = original._population(generation)
                    for row in rows:
                        proposal = json.loads(program(row.id))
                        policy = _policy_class(
                            proposal["name"], proposal["implementation"], proposal["description"]
                        )
                        row.name, row.description = policy.name, policy.description
                        row.implementation, row.policy_id = proposal["implementation"], policy.id
                        row.status, row.score, row.seed_scores = "evaluated", 7, {"0": 7}
                    if legacy:
                        original.elites = sorted(
                            original.elites + rows, key=lambda r: (-r.score, r.id)
                        )[: original.config.elite_size]
                        generation.elite_ids = [r.id for r in original.elites]
                        generation.status = "completed"
                    else:
                        original._promote(generation, rows)
                pending = Generation(number=3)
                original.generations.append(pending)
                original._population(pending)
                restored = self.agent()
                restored.restore(
                    [r.model_copy(deep=True) for r in original.organisms],
                    [g.model_copy(deep=True) for g in original.generations],
                )
                self.assertEqual([r.id for r in restored.elites], original.generations[1].elite_ids)
                self.assertEqual(restored.rng.getstate(), original.rng.getstate())
                rows = [r for r in restored.organisms if r.generation == 3]
                for row in rows:
                    row.score = 7
                restored._promote(restored.generations[-1], rows)
                edits = sorted(r.id for r in rows if r.kind == "edit")
                self.assertEqual(restored.generations[-1].elite_ids, edits)
                corrupted = [g.model_copy(deep=True) for g in original.generations]
                corrupted[0].elite_ids.reverse()
                with self.assertRaisesRegex(ValueError, "Checkpoint elites"):
                    self.agent().restore(
                        [r.model_copy(deep=True) for r in original.organisms], corrupted
                    )


if __name__ == "__main__":
    unittest.main()
