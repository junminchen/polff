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
Test: load ByteFF-Pol ForceField XML and print OpenMM potential energy.

Demonstrates single-molecule energy computation and multi-XML loading.

Examples
--------
# Single molecule (ethanol)
python test_energy.py --xml ethanol_files/ethanol_ff.xml \
                       --pdb ethanol_files/ethanol.pdb

# Two molecules in one system (requires combined PDB with both residues)
python test_energy.py --xml mol1_files/mol1_ff.xml mol2_files/mol2_ff.xml \
                       --pdb combined.pdb

# Verbose per-force breakdown (default) + CUDA platform
python test_energy.py --xml ethanol_files/ethanol_ff.xml \
                       --pdb ethanol_files/ethanol.pdb \
                       --platform CUDA
"""

import argparse

import openmm as mm
import openmm.app as app
import openmm.unit as unit

# IMPORTANT: import load_ff *before* creating any ForceField so that the
# ByteFF14Force parser is registered with app.ForceField.parsers.
from load_ff import load_forcefield


def compute_energy(xml_files: list, pdb_path: str,
                   platform_name: str = 'CPU') -> tuple:
    """Load a ForceField, create a gas-phase system, and report the energy.

    Parameters
    ----------
    xml_files     : list of *_ff.xml paths (one per molecule type)
    pdb_path      : PDB file whose residue names match the XML templates
    platform_name : OpenMM platform ('CPU', 'Reference', or 'CUDA')

    Returns
    -------
    total_energy_kcal : float
    per_force_kcal    : dict  {force_class_name: energy_kcal}
    """
    ff  = load_forcefield(*xml_files)
    pdb = app.PDBFile(pdb_path)

    # Gas-phase system (NoCutoff, no periodic boundary)
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)

    # Give every force its own group so we can query individual contributions
    force_names: dict = {}
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        f.setForceGroup(i)
        name = type(f).__name__
        force_names[i] = name

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    platform   = mm.Platform.getPlatformByName(platform_name)
    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    # ── total energy ──────────────────────────────────────────────────
    state = sim.context.getState(getEnergy=True)
    total_kcal = state.getPotentialEnergy().value_in_unit(
        unit.kilocalorie_per_mole)

    # ── per-force energies ────────────────────────────────────────────
    per_force: dict = {}
    for i in range(system.getNumForces()):
        s = sim.context.getState(getEnergy=True, groups={i})
        e = s.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        name = force_names[i]
        per_force[name] = per_force.get(name, 0.0) + e

    # ── report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PDB        : {pdb_path}")
    print(f"XML(s)     : {', '.join(xml_files)}")
    print(f"Platform   : {platform_name}")
    print(f"Atoms      : {system.getNumParticles()}")
    print(f"Forces     : {system.getNumForces()}")
    print(f"{'─'*60}")
    print(f"Total energy : {total_kcal:>12.4f}  kcal/mol")
    print(f"{'─'*60}")
    print("Per-force breakdown:")
    for name, e in per_force.items():
        print(f"  {name:<35s}  {e:>12.4f}  kcal/mol")
    print(f"{'='*60}\n")

    return total_kcal, per_force


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute OpenMM energy from ByteFF-Pol ForceField XML")
    parser.add_argument("--xml", nargs='+', required=True,
                        help="ForceField XML file(s) (*_ff.xml)")
    parser.add_argument("--pdb", required=True,
                        help="PDB file with matching residue names")
    parser.add_argument("--platform", default='CPU',
                        choices=['CPU', 'Reference', 'CUDA'],
                        help="OpenMM compute platform")
    args = parser.parse_args()

    compute_energy(args.xml, args.pdb, args.platform)
