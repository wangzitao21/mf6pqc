"""Fast scientific-invariant checks for the SaltLake_Brine3D example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "examples" / "SaltLake_Brine3D" / "case_config.py"
SPEC = importlib.util.spec_from_file_location("salt_lake_case_config", CONFIG_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"Could not load {CONFIG_PATH}")
case_config = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = case_config
SPEC.loader.exec_module(case_config)


class SaltLakeScientificConfigurationTests(unittest.TestCase):
    def test_bulk_volume_and_nominal_potassium_grade(self):
        fractions = case_config.BACKGROUND_MINERAL_VOLUME_FRACTIONS
        self.assertAlmostEqual(sum(fractions.values()), 0.70)
        self.assertAlmostEqual(
            case_config.INITIAL_POROSITY
            + case_config.INERT_SOLID_FRACTION
            + sum(fractions.values()),
            1.0,
        )
        self.assertAlmostEqual(
            case_config.potassium_grade_percent(fractions), 2.1981448924
        )

    def test_every_facies_preserves_total_evaporite_volume(self):
        grades = []
        for facies, fractions in case_config.FACIES_MINERAL_VOLUME_FRACTIONS.items():
            self.assertAlmostEqual(sum(fractions.values()), 0.70, msg=facies)
            moles = case_config.FACIES_MINERAL_MOLES[facies]
            volume = sum(
                moles[name]
                * case_config.MINERAL_MOLAR_VOLUMES_L_PER_MOL[name]
                for name in moles
            )
            self.assertAlmostEqual(volume, 0.70, msg=facies)
            grades.append(case_config.potassium_grade_percent(fractions))
        self.assertLess(grades[2], grades[0])
        self.assertGreater(grades[1], grades[0])

    def test_profiles_are_confined_three_layer_opposed_boundaries(self):
        for profile in case_config.PROFILES.values():
            self.assertEqual(profile.nlay, 3)
            self.assertEqual(profile.nxyz, np.prod(profile.shape))
            self.assertLess(profile.channel_column, profile.well_column)
            self.assertTrue(all(cell[0] == 0 for cell in profile.channel_cells))
            self.assertTrue(all(cell[0] == 0 for cell in profile.well_cells))
            self.assertEqual(
                profile.total_steps, profile.years * profile.steps_per_year
            )

    def test_heterogeneous_fields_match_modflow_cell_order(self):
        profile = case_config.PROFILES["smoke"]
        conductivity = case_config.initial_hydraulic_conductivity(profile)
        facies = case_config.kinetic_facies(profile)
        self.assertEqual(conductivity.shape, (profile.nxyz,))
        self.assertEqual(facies.shape, (profile.nxyz,))
        self.assertTrue(np.all(conductivity > 0.0))
        self.assertEqual(set(np.unique(facies)), {1, 2, 3})

    def test_highres_uses_reproducible_lognormal_random_field(self):
        profile = case_config.PROFILES["highres"]
        first = case_config.initial_hydraulic_conductivity(profile).reshape(
            profile.shape
        )
        second = case_config.initial_hydraulic_conductivity(profile).reshape(
            profile.shape
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(profile.nxyz, 5_400)
        self.assertTrue(profile.uniform_mineralogy)
        self.assertEqual(set(np.unique(case_config.kinetic_facies(profile))), {1})
        for layer, target in enumerate(
            profile.layer_geometric_mean_k_m_per_day
        ):
            self.assertAlmostEqual(
                float(np.exp(np.mean(np.log(first[layer])))),
                target,
                places=12,
            )
            self.assertGreater(float(np.std(np.log(first[layer]))), 0.70)


if __name__ == "__main__":
    unittest.main()
