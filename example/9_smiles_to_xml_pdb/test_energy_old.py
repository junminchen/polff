# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference energy test: load Gromacs parameters directly via AmoebaCalculator
and compare against the new ForceField XML approach.

Usage
-----
# Ethanol (ethanol_files must already contain params/ and ethanol.pdb)
python test_energy_old.py --top  ethanol_files/params/system_gas.top \
                           --json ethanol_files/params/ethanol.json \
                           --nbparams ethanol_files/params/ethanol_nb_params.json \
                           --name ethanol \
                           --pdb  ethanol_files/ethanol.pdb \
                           --xml  ethanol_files/ethanol_ff.xml
"""

import argparse
import json

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from byteff2.toolkit.openmmtool import AmoebaCalculator

# Register ByteFF14Force parser before any ForceField XML is loaded
from load_ff import load_forcefield


def run_old(top_file: str, mol_name: str,
            json_file: str, nbparams_file: str,
            pdb_path: str, platform_name: str = 'CPU') -> tuple:
    """Compute energy using the original AmoebaCalculator (Gromacs params)."""
    with open(json_file) as f:
        mol_params = json.load(f)
    with open(nbparams_file) as f:
        nb_meta = json.load(f)

    nonbonded_params = {mol_name: mol_params, 'metadata': nb_meta['metadata']}

    calc = AmoebaCalculator(top_file, nonbonded_params,
                             platform_name=platform_name,
                             separate_terms=True)

    pdb = app.PDBFile(pdb_path)
    positions = np.array([[v.x, v.y, v.z]
                           for v in pdb.positions.value_in_unit(unit.angstrom)])

    total_kcal, _ = calc._calculate_without_restraint(positions)
    per_force, _  = calc.get_separate_terms()
    return total_kcal, per_force


def run_new(xml_files: list, pdb_path: str,
            platform_name: str = 'CPU') -> tuple:
    """Compute energy using the new ForceField XML approach."""
    ff  = load_forcefield(*xml_files)
    pdb = app.PDBFile(pdb_path)

    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)

    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(i)

    force_names = {i: type(system.getForce(i)).__name__
                   for i in range(system.getNumForces())}

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    platform   = mm.Platform.getPlatformByName(platform_name)
    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    state = sim.context.getState(getEnergy=True)
    total_kcal = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)

    per_force: dict = {}
    for i in range(system.getNumForces()):
        s = sim.context.getState(getEnergy=True, groups={i})
        e = s.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        name = force_names[i]
        per_force[name] = per_force.get(name, 0.0) + e

    return total_kcal, per_force


def compare(top_file, mol_name, json_file, nbparams_file,
            pdb_path, xml_files, platform_name='CPU'):

    print(f"\n{'='*65}")
    print(f"Molecule  : {mol_name}")
    print(f"PDB       : {pdb_path}")
    print(f"TOP       : {top_file}")
    print(f"XML(s)    : {', '.join(xml_files)}")
    print(f"Platform  : {platform_name}")
    print(f"{'='*65}")

    # ── Reference (Gromacs → AmoebaCalculator) ────────────────────────
    e_old, pf_old = run_old(top_file, mol_name, json_file, nbparams_file,
                             pdb_path, platform_name)

    print(f"\n[OLD] AmoebaCalculator (direct Gromacs params)")
    print(f"  Total : {e_old:>12.4f}  kcal/mol")
    for name, e in pf_old.items():
        print(f"  {name:<35s}  {e:>10.4f}  kcal/mol")

    # ── New (ForceField XML) ──────────────────────────────────────────
    e_new, pf_new = run_new(xml_files, pdb_path, platform_name)

    print(f"\n[NEW] ForceField XML (app.ForceField)")
    print(f"  Total : {e_new:>12.4f}  kcal/mol")
    for name, e in pf_new.items():
        print(f"  {name:<35s}  {e:>10.4f}  kcal/mol")

    # ── Comparison ────────────────────────────────────────────────────
    diff = abs(e_new - e_old)
    status = "✓ MATCH" if diff < 1e-3 else f"✗ MISMATCH  Δ={diff:.6f}"
    print(f"\n{'─'*65}")
    print(f"  OLD total : {e_old:>12.4f}  kcal/mol")
    print(f"  NEW total : {e_new:>12.4f}  kcal/mol")
    print(f"  |Δ|       : {diff:>12.6f}  kcal/mol   {status}")
    print(f"{'='*65}\n")

    return e_old, e_new, diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare AmoebaCalculator vs ForceField XML energy")
    parser.add_argument("--top",      default="ethanol_files/params/system_gas.top",
                        help="Gromacs .top file (default: ethanol_files/params/system_gas.top)")
    parser.add_argument("--json",     default="ethanol_files/params/ethanol.json",
                        help="Per-atom params JSON (default: ethanol_files/params/ethanol.json)")
    parser.add_argument("--nbparams", default="ethanol_files/params/ethanol_nb_params.json",
                        help="Nonbonded-params JSON with metadata (default: ethanol_files/params/ethanol_nb_params.json)")
    parser.add_argument("--name",     default="ethanol",
                        help="Molecule name matching key in JSON (default: ethanol)")
    parser.add_argument("--pdb",      default="ethanol_files/ethanol.pdb",
                        help="PDB file with correct resname (default: ethanol_files/ethanol.pdb)")
    parser.add_argument("--xml",      nargs='+', default=["ethanol_files/ethanol_ff.xml"],
                        help="ForceField XML file(s) (default: ethanol_files/ethanol_ff.xml)")
    parser.add_argument("--platform", default='CPU',
                        choices=['CPU', 'Reference', 'CUDA'])
    args = parser.parse_args()

    compare(
        top_file=args.top,
        mol_name=args.name,
        json_file=args.json,
        nbparams_file=args.nbparams,
        pdb_path=args.pdb,
        xml_files=args.xml,
        platform_name=args.platform,
    )
