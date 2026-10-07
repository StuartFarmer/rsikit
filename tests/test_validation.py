"""Direct optimizer configuration has the same typed constraints as CLI input."""

import json
import unittest
from dataclasses import asdict, replace

from research.alphaevolve.original.agent import Config as AlphaConfig
from research.alphaevolve.paper.agent import Config as PaperConfig
from research.elitesearch.agent import Config as EliteConfig
from research.lineagesearch.agent import Config as LineageConfig
from research.shinkaevolve.agent import Config as ShinkaConfig


class ConfigValidationTests(unittest.TestCase):
    def test_direct_construction_and_replace_validate_types_and_bounds(self):
        for cls in (AlphaConfig, PaperConfig, EliteConfig, LineageConfig, ShinkaConfig):
            config = cls()
            self.assertEqual(
                asdict(replace(config, generation_concurrency=2))["generation_concurrency"], 2
            )
            for values in (
                {"generation_concurrency": 0},
                {"generation_concurrency": True},
                {"generation_concurrency": "2"},
                {"max_repairs": -1},
                {"generation_timeout": float("nan")},
            ):
                with self.subTest(config=cls, values=values), self.assertRaises(ValueError):
                    cls(**values)
                with self.subTest(replace=cls, values=values), self.assertRaises(ValueError):
                    replace(config, **values)

    def test_algorithm_constraints_and_disabled_features(self):
        for cls, values in (
            (AlphaConfig, {"mode": "unknown"}),
            (PaperConfig, {"reset_interval": 1}),
            (EliteConfig, {"new_fraction": 0.8, "remix_fraction": 0.4}),
            (EliteConfig, {"target_score": float("inf")}),
            (LineageConfig, {"cull_percent": 100}),
            (ShinkaConfig, {"parent_selection": "unknown"}),
            (ShinkaConfig, {"migration_rate": 2}),
        ):
            with self.subTest(config=cls, values=values), self.assertRaises(ValueError):
                cls(**values)
        self.assertEqual(AlphaConfig(proposals=0, meta_interval=0, reset_interval=0).proposals, 0)
        self.assertEqual(PaperConfig(migration_interval=0, migration_count=0).migration_count, 0)
        self.assertEqual(ShinkaConfig(generations=0, migration_interval=0).generations, 0)
        self.assertEqual(LineageConfig(max_attempts=0, bonus_batches=0).max_attempts, 0)
        # Elites accumulate across generations and can exceed one generation's population.
        self.assertEqual(EliteConfig(population_size=2, elite_size=10).elite_size, 10)

    def test_paper_feature_bounds_survive_json_configuration_roundtrip(self):
        config = PaperConfig(features={"speed": (0, 10, 5)})
        self.assertEqual(PaperConfig(**json.loads(json.dumps(asdict(config)))), config)
        for bounds in ([0, 10, True], ["0", 10, 5]):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                PaperConfig(features={"speed": bounds})
