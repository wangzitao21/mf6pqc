# GWE/VSC thermal coupling contract

MF6PQC implements MODFLOW 6 GWE and VSC through an explicit, opt-in coupling
method named `ThermalSNIA`. Existing `run()`, SNIA, SIA, and Strang cases do
not enter this code path.

## Operator order

For each TDIS step, ThermalSNIA performs:

1. expose the current GWE temperature and reaction-updated reference K to GWF;
2. solve every registered MODFLOW solution (GWF, component GWT models, GWE);
3. retain the VSC viscosity and effective K actually used by flow;
4. copy the post-GWE cell temperature to PhreeqcRM with `SetTemperature`;
5. run PhreeqcRM reactions for the same time step;
6. write reacted component concentrations to GWT;
7. update porosity, GWT MST porosity, GWE EST porosity, and reference K;
8. save aligned chemistry, thermal, viscosity, and K snapshots.

This is explicit sequential coupling. Temperature and chemistry feedback from
one completed step affect flow in the next step. The output therefore stores
both `temperature` (post-GWE and used by chemistry) and
`temperature_for_flow` (the lagged field used by VSC during that flow solve).

## Conductivity ownership

MODFLOW NPF K is hydraulic conductivity. Under VSC, two distinct fields must
not be conflated:

```text
reaction / porosity model -> KxxINPUT (reference hydraulic conductivity)
GWE temperature -> VSC viscosity -> Kxx (effective hydraulic conductivity)
```

For each cell MODFLOW applies:

```text
K_effective = K_reference * viscosity_reference / viscosity
```

MF6PQC writes only `K11INPUT`, `K22INPUT`, and `K33INPUT`. It never feeds a
VSC-adjusted K back into Kozeny-Carman, so the viscosity factor cannot
accumulate or be counted twice. Initial K22/K11 and K33/K11 anisotropy ratios
are preserved when reference K changes.

The following combinations are rejected:

- VSC plus `FluidAdjustedKozenyCarmanUpdater`;
- VSC plus MF6PQC's manual `boundary_conductance_updates`.

BUY may coexist with VSC because density and viscosity remain separately
owned by MODFLOW packages.

## Temperature and chemistry

The GWE dependent variable `X` is interpreted as degrees Celsius, matching
MODFLOW 6 GWE and PhreeqcRM `SetTemperature`. Before every thermal reaction
call MF6PQC validates this field and passes the complete cell-wise array to
PhreeqcRM.

Equilibrium thermodynamics then use the database's temperature dependence.
Custom kinetic `RATES` blocks must explicitly use `TK` (Kelvin) when an
Arrhenius or other temperature-dependent rate is intended. The bundled
`GWE_VSC_Reactive` example demonstrates this contract.

At MODFLOW initialization, MF6PQC checks that GWE IC temperature and EST
porosity match the fields already supplied to PhreeqcRM. This catches two
independently configured models that would otherwise start from inconsistent
states. The check can be configured but is enabled by default.

## Enabling the feature

Structured configuration is preferred:

```python
from mf6pqc import EnergyOptions, SimulationConfig

config = SimulationConfig(
    # existing fields omitted
    energy=EnergyOptions(
        enabled=True,
        viscosity_feedback=True,
        flow_model_name="gwf_model",
        energy_model_name="gwe_model",
    ),
)

simulator.run(method="ThermalSNIA")
```

The MODFLOW workspace must contain the matching GWE model, GWF-GWE exchange,
GWF VSC package linked to GWE `TEMPERATURE`, and one IMS solution that advances
GWE. Missing variables or incorrect model/package names fail with a specific
binding error before the first time step.

## Scope

The implemented direction is GWE temperature to flow viscosity and chemistry,
plus chemistry to porosity/reference K/thermal storage. Reaction enthalpy as a
source term back into GWE is not inferred automatically; it requires a
separate, explicitly parameterized heat-source contract.
