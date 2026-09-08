"""Exercise installed modflowapi wrappers without loading or running native solvers."""

import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import modflowapi
import numpy as np
from xmipy import XmiWrapper

from mf6pqc.backends import NativeBackendFactory
from mf6pqc.coupling.common import (
    cache_basic_geometry,
    cache_concentration_variables,
    write_concentrations_to_modflow,
)


class MemoryApi:
    """Small MODFLOW variable store; real modflowapi extension classes consume it."""

    def __init__(self):
        self.values = {
            "__INPUT__/SIM/NAM/SLNMNAMES": np.array(["SLN_1"]),
            "__INPUT__/SIM/NAM/SLNTYPE": np.array(["IMS6"]),
            "SLN_1/ID": np.array([1]),
            "SLN_1/MXITER": np.array([10]),
            "TDIS/NPER": np.array([1]),
        }
        for index, name in enumerate(("GWF", "GWT_NA_MODEL", "GWE"), 1):
            for variable, values in {
                "ID": [index],
                "IDSOLN": [1],
                "X": [1.0, 2.0],
                "DIS/NLAY": [1],
                "DIS/NROW": [1],
                "DIS/NCOL": [2],
                "DIS/NODES": [2],
                "DIS/TOP": [2.0, 3.0],
                "DIS/BOT": [0.0, 0.0],
                "DIS/AREA": [1.0, 1.0],
                "DIS/IDOMAIN": [1, 1],
            }.items():
                self.values[f"{name}/{variable}"] = np.array(values)

    def get_input_var_names(self):
        return tuple(self.values)

    def get_var_address(self, variable, component, package=None):
        return "/".join(p for p in (component, package, variable) if p).upper()

    def get_value(self, address):
        return self.values[address].copy()

    def get_value_ptr(self, address):
        return self.values[address]

    def get_grid_type(self, model_id):
        return "rectilinear"


class ModflowApiCompatibilityTests(unittest.TestCase):
    def test_native_factory_constructor_matches_installed_xmipy(self):
        # Run the real ModflowApi constructor, but stop at the native boundary.
        with patch.object(XmiWrapper, "__init__", autospec=True, return_value=None) as init:
            api = NativeBackendFactory().create_modflow_api("libmf6.dll", "workspace")
        api._state = None  # Native initialization was intentionally intercepted.
        self.assertIsInstance(api, modflowapi.ModflowApi)
        self.assertEqual(init.call_args.kwargs["working_directory"], "workspace")
        self.assertEqual(init.call_args.kwargs["logger_level"], 0)

    def test_coupling_calls_bind_to_installed_api_signatures(self):
        # Bind actual call sites rather than maintaining a second list of API methods.
        package = Path(__file__).resolve().parents[1] / "mf6pqc"
        checked = set()
        for path in package.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = node.func.value
                if not (
                    isinstance(owner, ast.Name)
                    and owner.id == "modflow_api"
                    or isinstance(owner, ast.Attribute)
                    and owner.attr == "modflow_api"
                ):
                    continue
                name = node.func.attr
                with self.subTest(file=path.name, method=name):
                    method = getattr(modflowapi.ModflowApi, name)
                    inspect.signature(method).bind(
                        object(),
                        *[object() for _ in node.args],
                        **{keyword.arg: object() for keyword in node.keywords},
                    )
                checked.add(name)
        self.assertTrue({"get_value_ptr", "solve", "prepare_time_step", "finalize"} <= checked)

    def test_real_extensions_load_models_and_preserve_geometry_and_live_arrays(self):
        api = MemoryApi()
        simulation = NativeBackendFactory().load_modflow_simulation(api)
        self.assertEqual(set(simulation.model_names), {"gwf", "gwt_na_model", "gwe"})
        self.assertEqual(simulation.solutions[1].mxiter, 10)
        sim = SimpleNamespace(modflow_api=api, sim=simulation, flow_model_name="GWF", nxyz=2)
        cache_basic_geometry(sim)
        np.testing.assert_array_equal(sim.cell_thick, [2.0, 3.0])
        for name in ("GWF", "GWT_NA_MODEL", "GWE"):
            np.testing.assert_array_equal(simulation.get_model(name).X.ravel(), [1.0, 2.0])
        variables = cache_concentration_variables(api, ["Na"], nxyz=2)
        write_concentrations_to_modflow(variables, [slice(0, 2)], np.array([3.0, 4.0]))
        np.testing.assert_array_equal(simulation.get_model("GWT_NA_MODEL").X.ravel(), [3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
