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
Helper module for loading ByteFF-Pol ForceField XML files.

The ForceField XML produced by generate_ff_xml_pdb.py uses a custom XML tag
<ByteFF14Force> to store 1-4 VdW correction parameters. This module registers
the corresponding ForceField generator *before* app.ForceField parses the XML.

Usage
-----
    from load_ff import load_forcefield
    import openmm.app as app

    # Single molecule
    ff = load_forcefield("ethanol_files/ethanol_ff.xml")

    # Multiple molecules simultaneously (types are globally unique, no conflict)
    ff = load_forcefield("ethanol_files/ethanol_ff.xml",
                         "methanol_files/methanol_ff.xml")

    pdb    = app.PDBFile("system.pdb")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
"""

import openmm as mm
import openmm.app as app
from openmm.app.forcefield import parsers


class ByteFF14Generator:
    """ForceField generator for the <ByteFF14Force> XML tag.

    Creates a CustomBondForce that applies 1-4 (×0.5) and 1-5 (×1.0) VdW
    corrections without the charge-transfer term.
    Atom types are globally unique integers (one type per atom), so each
    <Bond type1="..." type2="..."/> entry maps to exactly one atom pair in the
    system regardless of how many molecules are present.
    """

    def __init__(self, ff):
        self.ff = ff
        self.energy = ""
        self.per_bond_params: list = []
        self.bond_entries: list = []   # [(type1_str, type2_str, [param_values])]

    @staticmethod
    def parseElement(element, ff):
        # Re-use the same generator when loading multiple XML files so that
        # 1-4 bonds from all molecules end up in a single CustomBondForce.
        existing = [g for g in ff._forces if isinstance(g, ByteFF14Generator)]
        gen = existing[0] if existing else ByteFF14Generator(ff)
        if not existing:
            ff.registerGenerator(gen)

        if not gen.energy:
            gen.energy = element.attrib['energy']

        if not gen.per_bond_params:
            for p in element.findall('PerBondParameter'):
                gen.per_bond_params.append(p.attrib['name'])

        for bond in element.findall('Bond'):
            t1 = bond.attrib['type1']
            t2 = bond.attrib['type2']
            vals = [float(bond.attrib[p]) for p in gen.per_bond_params]
            gen.bond_entries.append((t1, t2, vals))

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        force = mm.CustomBondForce(self.energy)
        sys.addForce(force)
        for p in self.per_bond_params:
            force.addPerBondParameter(p)

        # Build a type → atom-index lookup (O(N), then O(M) bond creation)
        type_to_idx: dict = {}
        for i, atom in enumerate(data.atoms):
            type_to_idx[data.atomType[atom]] = i

        for t1, t2, params in self.bond_entries:
            idx1 = type_to_idx.get(t1)
            idx2 = type_to_idx.get(t2)
            if idx1 is not None and idx2 is not None:
                force.addBond(idx1, idx2, params)


# Register once at import time so the parser is ready when ForceField loads XML
if 'ByteFF14Force' not in parsers:
    parsers['ByteFF14Force'] = ByteFF14Generator.parseElement


def load_forcefield(*xml_files: str) -> app.ForceField:
    """Load one or more ByteFF-Pol ForceField XML files into an app.ForceField.

    The ByteFF14Force parser is registered automatically. Multiple XMLs for
    different molecules can be combined safely because atom type IDs are
    deterministic 64-bit hashes of (mol_name, atom_index), guaranteeing global
    uniqueness.

    Parameters
    ----------
    *xml_files : paths to *_ff.xml files from generate_ff_xml_pdb.py

    Returns
    -------
    app.ForceField ready to call createSystem() on a matching topology.
    """
    return app.ForceField(*xml_files)
