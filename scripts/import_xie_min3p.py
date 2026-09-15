"""Reproduce Xie B2--B4 chemistry from the supplied MIN3P benchmark database.

Usage: python scripts/import_xie_min3p.py PATH_TO_ORIGINAL_INPUT_DIRECTORY
This is a benchmark input translator, not a kinetic solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {2: "ex015_Xie2015_B2", 3: "ex016_Xie2015_B3", 4: "ex017_Xie2015_B4"}
AQUEOUS = {
    "h+1": "H+",
    "h2o": "H2O",
    "o2(aq)": "O2",
    "h2(aq)": "H2",
    "ca+2": "Ca+2",
    "co3-2": "CO3-2",
    "so4-2": "SO4-2",
    "na+1": "Na+",
    "fe+2": "Fe+2",
    "fe+3": "Fe+3",
    "al+3": "Al+3",
    "k+1": "K+",
    "khso4(aq)": "KHSO4",
    "h2so4(aq)": "H2SO4",
    "feco3+": "FeCO3+",
    "oh-": "OH-",
    "caoh+": "CaOH+",
    "cahco3+": "CaHCO3+",
    "caco3aq": "CaCO3",
    "caso4aq": "CaSO4",
    "cahso4+": "CaHSO4+",
    "naco3-": "NaCO3-",
    "nahco3aq": "NaHCO3",
    "naso4-": "NaSO4-",
    "kso4-": "KSO4-",
    "aloh+2": "AlOH+2",
    "al(oh)2+": "Al(OH)2+",
    "al(oh)4-": "Al(OH)4-",
    "also4+": "AlSO4+",
    "alhso42+": "AlHSO4+2",
    "al(so4)2-": "Al(SO4)2-",
    "al(oh)3aq": "Al(OH)3",
    "feoh+": "FeOH+",
    "feoh3-1": "Fe(OH)3-",
    "feso4aq": "FeSO4",
    "fehso4+": "FeHSO4+",
    "fehco3+": "FeHCO3+",
    "feco3aq": "FeCO3",
    "feoh2aq": "Fe(OH)2",
    "feoh+2": "FeOH+2",
    "feso4+": "FeSO4+",
    "fehso42+": "FeHSO4+2",
    "feoh2+": "Fe(OH)2+",
    "feoh3aq": "Fe(OH)3",
    "feoh4-": "Fe(OH)4-",
    "fe(so4)2-": "Fe(SO4)2-",
    "fe2(oh)2+4": "Fe2(OH)2+4",
    "fe3(oh)4+5": "Fe3(OH)4+5",
    "hco3-": "HCO3-",
    "h2co3aq": "H2CO3",
    "hso4-": "HSO4-",
}
MINERALS = {
    "calcite": ("Calcite", "CaCO3", {"Ca": 1, "C": 1, "O": 3}),
    "gypsum": ("Gypsum", "CaSO4:2H2O", {"Ca": 1, "S": 1, "O": 6, "H": 4}),
    "ferrihydrite": ("Ferrihydrite", "Fe(OH)3", {"Fe": 1, "O": 3, "H": 3}),
    "jarositek": ("Jarosite", "KFe3(SO4)2(OH)6", {"K": 1, "Fe": 3, "S": 2, "O": 14, "H": 6}),
    "gibbsite(c)": ("Gibbsite", "Al(OH)3", {"Al": 1, "O": 3, "H": 3}),
    "siderite(d)": ("Siderite", "FeCO3", {"Fe": 1, "C": 1, "O": 3}),
}


def records(path):
    return [
        shlex.split(line.split(";", 1)[0])
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("!")
    ]


def amount(value):
    return float(str(value).lower().replace("d", "e"))


def section(lines, name, count=False):
    i = lines.index([name]) + 1
    if count:
        return [v[0] for v in lines[i + 1 : i + 1 + int(lines[i][0])]]
    return lines[i]


def equation(product, terms, dissolution=False):
    left, right = ([product], []) if dissolution else ([], [product])
    for name, coefficient in terms:
        side = right if (coefficient > 0) == dissolution else left
        side.append(f"{abs(coefficient):g} {AQUEOUS[name]}")
    return " + ".join(left) + " = " + " + ".join(right)


def gamma(charge, radius, b):
    if abs(charge) < 1e-10:
        return "    -gamma 0 0.1"
    if radius:
        return f"    -gamma {radius:g} {b:g}"
    # MIN3P Davies uses b=0.24, rather than PHREEQC's default 0.3.
    # Extended Debye-Huckel reproduces its functional form at 25 C.
    return f"    -gamma {1 / 0.3281:.10g} {0.5115 * charge * charge * 0.24:.10g}"


def generate(source, level):
    dat = source / f"porperm_benchmark_level{level}_min3p.dat"
    lines = records(dat)
    components = section(lines, "components", True)
    complexes = section(lines, "secondary aqueous species", True)
    minerals = section(lines, "minerals", True)
    comp = {r[0]: r[1:] for r in records(source / "database" / "comp.dbs") if r != ["end"]}
    rows = records(source / "database" / "complex.dbs")
    aqueous = {rows[i][0]: (rows[i][1:], rows[i + 1]) for i in range(0, len(rows) - 1, 2)}
    rows = records(source / "database" / "mineral.dbs")
    mineral_db = {rows[i][0]: rows[i + 1 : i + 5] for i in range(0, len(rows) - 1, 5)}
    out = [
        "# Generated from the supplied MIN3P benchmark database; see min3p_parameters.json.",
        "# Only the original aqueous network is enabled. Temperature: 25 C.",
        "SOLUTION_MASTER_SPECIES",
        "H H+ -1 H 1.008",
        "H(1) H+ -1 H",
        "H(0) H2 0 H",
        "E e- 0 0 0",
        "O H2O 0 O 15.9999",
        "O(-2) H2O 0 O",
        "O(0) O2 0 O",
        "Ca Ca+2 0 Ca 40.080",
        "C CO3-2 2 HCO3 12.0114",
        "C(4) CO3-2 2 HCO3",
        "S SO4-2 0 SO4 32.064",
        "S(6) SO4-2 0 SO4",
        "Na Na+ 0 Na 22.9898",
        "Alkalinity CO3-2 1 Ca0.5(CO3)0.5 50.0447",
    ]
    if level >= 3:
        out += [
            "Fe Fe+2 0 Fe 55.847",
            "Fe(2) Fe+2 0 Fe",
            "Fe(3) Fe+3 -2 Fe",
            "Al Al+3 0 Al 26.9815",
            "K K+ 0 K 39.102",
        ]
    out += [
        "SOLUTION_SPECIES",
        "e- = e-",
        "H2O = H2O",
        "2 H2O = O2 + 4 H+ + 4 e-",
        "    -log_k -86.08",
        "    -gamma 0 0.1",
        "2 H+ + 2 e- = H2",
        "    -log_k -3.067",
        "    -gamma 0 0.1",
    ]
    for c in components:
        if c in {"o2(aq)", "fe+3"}:
            continue
        vals = list(map(float, comp[c]))
        out += [f"{AQUEOUS[c]} = {AQUEOUS[c]}", gamma(*vals[:3])]
    if level >= 3:
        out += ["Fe+2 = Fe+3 + e-", "    -log_k -13.0475", "    -gamma 9 0"]
    for c in complexes:
        if c == "h2(aq)":
            continue
        header, r = aqueous[c]
        enthalpy, logk, charge, radius, b = map(float, header[:5])
        terms = [(r[j], float(r[j + 1])) for j in range(1, len(r), 2)]
        out += [
            equation(AQUEOUS[c], terms),
            f"    -log_k {logk:g}",
            f"    -delta_h {enthalpy:g} kcal",
            gamma(charge, radius, b),
        ]
    out.append("PHASES")
    entries = []
    # The interior/background mineral block is used for all active rock cells.
    starts = [i for i, r in enumerate(lines) if r == ["mineral input"]]
    mi = starts[1 if level == 4 else 0] + 1
    for j, mineral in enumerate(minerals):
        _, massdensity, r, equilibrium = mineral_db[mineral]
        mass, density = map(float, massdensity)
        name, formula, stoich = MINERALS[mineral]
        terms = [(r[k], float(r[k + 1])) for k in range(1, len(r), 2)]
        logk, enthalpy = map(float, equilibrium[1:])
        out += [
            name,
            "    " + equation(formula, terms, True),
            f"    -log_k {-logk:g}",
            f"    -delta_h {-enthalpy:g} kcal",
        ]
        initial = amount(lines[mi + 2 * j][0])
        is_equilibrium = lines[mi + 2 * j][1]
        if is_equilibrium != ".false.":
            raise ValueError("This translator expects the original kinetic benchmarks")
        power = {"constant": 0.0, "twothird": 2 / 3}[lines[mi + 2 * j][2]]
        entries.append(
            dict(
                name=name,
                stoichiometry=stoich,
                molar_volume_l_per_mol=mass / (1000 * density),
                initial_volume_fraction=initial,
                initial_amount_mol_bulk=max(initial, amount(lines[mi + 2 * j + 1][0]))
                * 1000
                * density
                / mass,
                minimum_amount_mol_bulk=amount(lines[mi + 2 * j + 1][0]) * 1000 * density / mass,
                rate_constant_mol_bulk_per_second=amount(lines[mi + 2 * j + 1][1]),
                surface_exponent=power,
            )
        )
    out += ["O2(g)", "    O2 = O2", "    -log_k -2.898", "END"]
    pqi = [
        "# MIN3P 'free' means TOTAL component concentration (manual Block 14).",
        "# Fixed pH is not adjusted for charge balance; initial B3/B4 Na is.",
        "KNOBS",
        "    -convergence_tolerance 1e-12",
        "    -iterations 300",
        "END",
    ]
    solution_totals = {}
    for number, block in [
        (0, "initial condition - reactive transport"),
        (1, "boundary conditions - reactive transport"),
    ]:
        start = lines.index([block])
        ci = (
            next(i for i in range(start + 1, len(lines)) if lines[i] == ["concentration input"]) + 1
        )
        vals = dict(zip(components, lines[ci : ci + len(components)], strict=True))
        solution_totals[str(number)] = {
            c: {"value": amount(value[0]), "constraint": value[1]} for c, value in vals.items()
        }
        pqi += [
            f"SOLUTION {number}",
            "    units mol/kgw",
            "    temp 25",
            f"    pH {vals['h+1'][0]}",
        ]
        for compname, phname in [
            ("ca+2", "Ca"),
            ("co3-2", "C(4)"),
            ("so4-2", "S(6)"),
            ("na+1", "Na"),
            ("al+3", "Al"),
            ("k+1", "K"),
            ("fe+2", "Fe(2)"),
            ("fe+3", "Fe(3)"),
        ]:
            if compname in vals:
                v, kind = vals[compname]
                charge = " charge" if kind == "charge" else ""
                pqi.append(f"    {phname} {amount(v):.16g}{charge}")
        if level >= 3:
            v, kind = vals["o2(aq)"]
            if kind == "pe":
                pe = amount(v)
                log_o2 = 4 * (pe + amount(vals["h+1"][0])) - 86.08
                pqi += [f"    pe {pe:g}", f"    O(0) {2 * 10**log_o2:.16g}"]
            elif kind == "po2":
                import math

                logpo2 = math.log10(amount(v))
                pe = 21.52 - amount(vals["h+1"][0]) + (logpo2 - 2.898) / 4
                pqi += [f"    pe {pe:.16g}", f"    O(0) 1 O2(g) {logpo2:g}"]
            else:
                raise ValueError(kind)
        pqi.append("END")
    pqi.append("KINETICS 1")
    for e in entries:
        pqi += [
            e["name"],
            f"    -m0 {e['initial_amount_mol_bulk']:.17g}",
            "    -tol 1e-10",
            f"    -parms {e['rate_constant_mol_bulk_per_second']:.17g} {e['minimum_amount_mol_bulk']:.17g}",
        ]
    pqi += ["    -cvode true", "    -cvode_steps 1000", "    -bad_step_max 1000", "END"]
    pqi.append("RATES")
    for e in entries:
        pqi += [e["name"], "    -start", "10 k = PARM(1)"]
        if e["surface_exponent"]:
            pqi.append("20 IF M > 0 THEN k = k * (M/M0)^(2/3) ELSE k = 0")
        pqi += [
            f'30 moles = k * (1 - SR("{e["name"]}")) * TIME',
            "40 IF moles > 0 AND moles > M-PARM(2) THEN moles = M-PARM(2)",
            "50 SAVE moles",
            "    -end",
        ]
    names = [e["name"] for e in entries]
    headings = [s for n in names for s in (n, "d_" + n, "SR_" + n, "SI_" + n)]
    pqi += ["END", "USER_PUNCH 1", "    -headings " + " ".join(headings), "    -start"]
    for j, n in enumerate(names):
        pqi.append(f'{10 * (j + 1)} PUNCH KIN("{n}"), KIN_DELTA("{n}"), SR("{n}"), SI("{n}")')
    pqi += [
        "    -end",
        "END",
        "SELECTED_OUTPUT 1",
        "    -reset false",
        "    -high_precision true",
        "PRINT",
        "    -warnings 0",
        "END",
    ]
    files = [dat, *(source / "database").glob("*.dbs")]
    settings = dict(
        level=level,
        source_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        minerals=entries,
        solutions=solution_totals,
        porosity=0.35,
        hydraulic_conductivity_m_per_s=1.16e-4,
        diffusion_m2_per_s=1e-9 if level == 4 else 0.0,
        final_years={2: 1000, 3: 300, 4: 3000}[level],
        original_max_step_years=0.1,
        spatial_weighting="upstream",
        temperature_c=25,
        reference_times_years={2: [10, 100, 1000], 3: [10, 100, 300], 4: [100, 1000, 3000]}[level],
    )
    target = ROOT / "examples" / CASES[level] / "input_data"
    (target / "database.dat").write_text("\n".join(out) + "\n", encoding="utf-8")
    (target / "input.pqi").write_text("\n".join(pqi) + "\n", encoding="utf-8")
    (target / "min3p_parameters.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    for level in CASES:
        generate(args.source, level)
