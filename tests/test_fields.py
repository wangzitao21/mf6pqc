from __future__ import annotations

import unittest

import numpy as np

from mf6pqc.input_processing import create_ic_array_from_map, setup_mixed_ic
from mf6pqc.output_processing import (
    extract_output_information,
    update_diffc,
    update_porosity,
)
from mf6pqc.permeability import (
    DensityCoupledKozenyCarmanUpdater,
    KozenyCarmanUpdater,
    PowerLawUpdater,
)
from mf6pqc.utils import ensure_array, get_gwt_model_name, get_species_slice


class RecordingChemistry:
    def __init__(self) -> None:
        self.mixed = None

    def InitialPhreeqc2Module_mix(self, first, second, fractions) -> None:
        self.mixed = (first.copy(), second.copy(), fractions.copy())


class ArrayUtilityTests(unittest.TestCase):
    def test_scalar_is_expanded_without_aliasing(self) -> None:
        actual = ensure_array(3, "porosity", 0.25)
        np.testing.assert_array_equal(actual, [0.25, 0.25, 0.25])
        self.assertEqual(actual.dtype, np.dtype(float))

    def test_array_is_flattened_and_copied(self) -> None:
        source = np.array([[1.0, 2.0]])
        actual = ensure_array(2, "field", source)
        source[0, 0] = 99.0
        np.testing.assert_array_equal(actual, [1.0, 2.0])

    def test_wrong_cell_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match nxyz"):
            ensure_array(3, "porosity", [0.2, 0.3])

    def test_species_slice_uses_component_major_layout(self) -> None:
        self.assertEqual(get_species_slice(4, 2), slice(8, 12))

    def test_long_modflow_names_are_short_and_collision_resistant(self) -> None:
        first = get_gwt_model_name("VeryLongComponentAlpha")
        second = get_gwt_model_name("VeryLongComponentBeta")
        self.assertLessEqual(len(first), 16)
        self.assertLessEqual(len(second), 16)
        self.assertNotEqual(first, second)


class InitialConditionTests(unittest.TestCase):
    def test_module_map_is_packed_in_phreeqcrm_order(self) -> None:
        packed = create_ic_array_from_map(
            2,
            {
                "solution": [3, 4],
                "equilibrium_phases": 7,
                "kinetics": [8, 9],
            },
        )
        self.assertEqual(packed.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(packed[0:2], [3, 4])
        np.testing.assert_array_equal(packed[2:4], [7, 7])
        np.testing.assert_array_equal(packed[12:14], [8, 9])
        np.testing.assert_array_equal(packed[4:12], -1)

    def test_mixing_fraction_is_repeated_for_all_seven_modules(self) -> None:
        chemistry = RecordingChemistry()
        setup_mixed_ic(
            chemistry,
            2,
            {"solution": 1},
            {"solution": 2},
            [0.25, 0.75],
        )
        self.assertIsNotNone(chemistry.mixed)
        np.testing.assert_array_equal(
            chemistry.mixed[2], np.tile([0.25, 0.75], 7)
        )


class FeedbackTests(unittest.TestCase):
    def test_mineral_delta_headings_drive_porosity_change(self) -> None:
        indices, volumes, names = extract_output_information(
            ["Ca", "d_Calcite", "Calcite", "d_Gypsum"],
            {"Calcite": 0.04, "Gypsum": 0.07},
        )
        np.testing.assert_array_equal(indices, [1, 3])
        np.testing.assert_array_equal(names, ["Calcite", "Gypsum"])
        selected = np.array(
            [
                [0.0, 0.0],
                [0.10, -0.05],
                [0.0, 0.0],
                [0.20, 0.10],
            ]
        )
        actual = update_porosity(selected, indices, volumes, np.array([0.3, 0.3]))
        np.testing.assert_allclose(actual, [0.282, 0.295])

    def test_porosity_is_clipped_to_physical_bounds(self) -> None:
        actual = update_porosity(
            np.array([[10.0, -100.0]]),
            np.array([0]),
            np.array([[1.0]]),
            np.array([0.2, 0.2]),
        )
        np.testing.assert_allclose(actual, [1.0e-4, 1.0])

    def test_diffusion_uses_porosity_tortuosity_factor(self) -> None:
        np.testing.assert_allclose(
            update_diffc(np.array([0.125, 1.0]), np.array([2.0, 3.0])),
            [1.0, 3.0],
        )

    def test_kozeny_carman_preserves_k_when_porosity_is_unchanged(self) -> None:
        updater = KozenyCarmanUpdater()
        k = np.array([1.0, 4.0])
        phi = np.array([0.2, 0.35])
        np.testing.assert_allclose(updater.update(k, phi, phi), k)

    def test_density_viscosity_updater_applies_documented_factors(self) -> None:
        updater = DensityCoupledKozenyCarmanUpdater()
        actual = updater.update(
            np.array([2.0]),
            np.array([0.25]),
            np.array([0.25]),
            density_old=np.array([1.0]),
            density_new=np.array([1.1]),
            viscosity_old=np.array([1.0]),
            viscosity_new=np.array([1.25]),
        )
        np.testing.assert_allclose(actual, [1.76])

    def test_power_law_exponent_is_configurable(self) -> None:
        actual = PowerLawUpdater(n=3.0).update(
            np.array([2.0]), np.array([0.2]), np.array([0.4])
        )
        np.testing.assert_allclose(actual, [16.0])


if __name__ == "__main__":
    unittest.main()

