"""
Generate OpenMM ForceField-compatible XML and PDB from SMILES.

Unlike generate_xml_pdb.py (which produces a serialized System XML), this script
produces a *ForceField XML* loadable with app.ForceField, enabling:
  - Topology-aware system creation
  - Multi-molecule systems (load several XMLs simultaneously)
  - Per-residue template matching

Usage:
    python generate_ff_xml_pdb.py --smiles "CCO" --name ethanol --output ethanol_files

Then test with:
    python test_energy.py --xml ethanol_files/ethanol_ff.xml \
                           --pdb ethanol_files/ethanol.pdb

Force field XML structure:
  - HarmonicBondForce / HarmonicAngleForce / PeriodicTorsionForce  (bonded)
  - AmoebaMultipoleForce                                            (polarizable charges)
  - CustomNonbondedForce   bondCutoff=3  (VdW; 1-5 at full scale)
  - ByteFF14Force          (1-4 VdW at 0.5 scale, handled by load_ff.py)
"""

import collections
import hashlib
import io
import os
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import openmm as omm
import openmm.app as app
import openmm.unit as unit
from openmm.app.gromacstopfile import GromacsTopFile

from byteff2.toolkit.protocol import Protocol
from byteff2.toolkit.openmmtool import nx_covalent_map_and_pairs


# ── Helpers ───────────────────────────────────────────────────────────────────

def atom_type_int(mol_name: str, atom_idx: int) -> str:
    """Return a deterministic, globally unique integer string for this atom type.

    Uses a 64-bit MD5 hash so different molecules never collide, and the value
    is always a positive integer (required by AmoebaMultipoleForce's XML parser
    which calls int() on every atom type encountered during axis assignment).
    """
    h = hashlib.md5(f"{mol_name}@{atom_idx}".encode()).hexdigest()
    return str(int(h[:16], 16) + 1)   # +1 ensures non-zero


def resname_from_mol(mol_name: str) -> str:
    """Generate a 3-char uppercase PDB residue name from a molecule name."""
    letters = ''.join(c for c in mol_name if c.isalpha()).upper()
    return (letters + "LIG")[:3]


# ── Core XML writer ───────────────────────────────────────────────────────────

def generate_ff_xml(mol_name: str, output_path: str,
                    top: GromacsTopFile, bonded_system: omm.System,
                    nonbonded_params: dict) -> None:
    """Write a ForceField-compatible XML for a single molecule.

    Parameters
    ----------
    mol_name         : molecule key in nonbonded_params
    output_path      : where to write the XML
    top              : GromacsTopFile (provides topology / atom info)
    bonded_system    : omm.System from top.createSystem() (bonded forces only)
    nonbonded_params : dict returned by Protocol.generate_ff_params()
    """
    params = nonbonded_params[mol_name]
    metadata = nonbonded_params.get('metadata', {})
    s12 = metadata.get('s12', 0.15)
    disp_damping = metadata.get('disp_damping', 0.4)

    resname = resname_from_mol(mol_name)

    # ── collect atom info from topology ──────────────────────────────
    residues = list(top.topology.residues())
    atoms_list = list(residues[0].atoms())   # single residue for a gas-phase mol
    natoms = len(atoms_list)

    atom_names = [a.name for a in atoms_list]
    atom_elems = [a.element.symbol if a.element else 'C' for a in atoms_list]
    atom_masses = [a.element.mass.value_in_unit(unit.dalton)
                   if a.element else 12.0 for a in atoms_list]

    atom_to_idx = {a: i for i, a in enumerate(atoms_list)}
    bonds = [(atom_to_idx[b[0]], atom_to_idx[b[1]])
             for b in top.topology.bonds()
             if b[0] in atom_to_idx and b[1] in atom_to_idx]

    # 1-4 pairs for the correction force
    _, pairs = nx_covalent_map_and_pairs(natoms, bonds)

    # Unique integer type IDs per atom
    type_ids = [atom_type_int(mol_name, i) for i in range(natoms)]

    root = ET.Element("ForceField")

    # ── AtomTypes ─────────────────────────────────────────────────────
    at_elem = ET.SubElement(root, "AtomTypes")
    for i in range(natoms):
        ET.SubElement(at_elem, "Type",
                      name=type_ids[i],
                      **{"class": type_ids[i]},
                      element=atom_elems[i],
                      mass=f"{atom_masses[i]:.4f}")

    # ── Residues ──────────────────────────────────────────────────────
    res_elem = ET.SubElement(root, "Residues")
    res = ET.SubElement(res_elem, "Residue", name=resname)
    for i in range(natoms):
        ET.SubElement(res, "Atom", name=atom_names[i], type=type_ids[i])
    for i, j in bonds:
        ET.SubElement(res, "Bond",
                      atomName1=atom_names[i], atomName2=atom_names[j])

    # ── HarmonicBondForce ─────────────────────────────────────────────
    hbf_elem = ET.SubElement(root, "HarmonicBondForce")
    for fi in range(bonded_system.getNumForces()):
        force = bonded_system.getForce(fi)
        if isinstance(force, omm.HarmonicBondForce):
            for bi in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(bi)
                ET.SubElement(hbf_elem, "Bond",
                              type1=type_ids[p1], type2=type_ids[p2],
                              length=f"{length.value_in_unit(unit.nanometer):.9f}",
                              k=f"{k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2):.4f}")
            break

    # ── HarmonicAngleForce ────────────────────────────────────────────
    haf_elem = ET.SubElement(root, "HarmonicAngleForce")
    for fi in range(bonded_system.getNumForces()):
        force = bonded_system.getForce(fi)
        if isinstance(force, omm.HarmonicAngleForce):
            for ai in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(ai)
                ET.SubElement(haf_elem, "Angle",
                              type1=type_ids[p1], type2=type_ids[p2],
                              type3=type_ids[p3],
                              angle=f"{angle.value_in_unit(unit.radian):.10f}",
                              k=f"{k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2):.4f}")
            break

    # ── PeriodicTorsionForce ──────────────────────────────────────────
    # Multiple terms with the same atom quadruplet are merged into one entry
    torsion_groups: dict = collections.defaultdict(list)
    for fi in range(bonded_system.getNumForces()):
        force = bonded_system.getForce(fi)
        if isinstance(force, omm.PeriodicTorsionForce):
            for ti in range(force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(ti)
                torsion_groups[(p1, p2, p3, p4)].append((
                    per,
                    phase.value_in_unit(unit.radian),
                    k.value_in_unit(unit.kilojoule_per_mole)
                ))
            break

    ptf_elem = ET.SubElement(root, "PeriodicTorsionForce")
    for (p1, p2, p3, p4), terms in torsion_groups.items():
        attribs = {
            "type1": type_ids[p1], "type2": type_ids[p2],
            "type3": type_ids[p3], "type4": type_ids[p4],
        }
        for n, (per, phase, k) in enumerate(terms, 1):
            attribs[f"periodicity{n}"] = str(per)
            attribs[f"phase{n}"] = f"{phase:.10f}"
            attribs[f"k{n}"] = f"{k:.6f}"
        ET.SubElement(ptf_elem, "Proper", **attribs)

    # ── AmoebaMultipoleForce ──────────────────────────────────────────
    # kz="0" kx="0"  → NoAxisType (each atom is a monopole/dipole with no
    # local frame needed since dipoles/quadrupoles are zero).
    # No pgrpX attributes → each atom is its own polarization group; the
    # ForceField generator infers Polarization12/13/14 from bond topology.
    amp_elem = ET.SubElement(root, "AmoebaMultipoleForce",
                              direct11Scale="0.0",  direct12Scale="1.0",
                              direct13Scale="1.0",  direct14Scale="1.0",
                              mpole12Scale="0.0",   mpole13Scale="0.0",
                              mpole14Scale="0.5",   mpole15Scale="0.8",
                              mutual11Scale="1.0",  mutual12Scale="1.0",
                              mutual13Scale="1.0",  mutual14Scale="1.0",
                              polar12Scale="0.0",   polar13Scale="0.0",
                              polar14Intra="0.5",   polar14Scale="1.0",
                              polar15Scale="1.0")
    for i in range(natoms):
        ET.SubElement(amp_elem, "Multipole",
                      type=type_ids[i],
                      kz="0", kx="0",
                      c0=f"{params['charge'][i]:.8f}",
                      d1="0.0", d2="0.0", d3="0.0",
                      q11="0.0", q21="0.0", q22="0.0",
                      q31="0.0", q32="0.0", q33="0.0")
    for i in range(natoms):
        # polarizability == pol_damping for ByteFF-Pol, so pdamp = alpha^(1/6)
        ET.SubElement(amp_elem, "Polarize",
                      type=type_ids[i],
                      polarizability=f"{params['alpha'][i]:.10e}",
                      thole="0.3900")

    # ── CustomNonbondedForce ──────────────────────────────────────────
    # bondCutoff=4: excludes 1-2, 1-3, 1-4, and 1-5 pairs.
    # 1-4 (×0.5) and 1-5 (×1.0) are added back via ByteFF14Force below,
    # intentionally WITHOUT the charge-transfer (CTE/CTL) correction term,
    # matching the behaviour of generate_openmm_system / AmoebaCalculator.
    func = (
        f"6*A*exp(B*(1-r/rvdw))-C6/rvdw^6/({disp_damping}+(r/rvdw)^6)"
        f"+({s12}/r)^12-CTE*exp(-(CTL*r/rvdw)^3)/r^4"
        f"; A=sqrt(A1*A2); B=sqrt(B1*B2); rvdw=(rvdw1+rvdw2)/2"
        f"; C6=sqrt(C61*C62); CTE=sqrt(CTE1*CTE2); CTL=sqrt(CTL1*CTL2)"
    )
    cnf_elem = ET.SubElement(root, "CustomNonbondedForce",
                              energy=func, bondCutoff="4")
    for pname in ["A", "B", "C6", "rvdw", "CTE", "CTL"]:
        ET.SubElement(cnf_elem, "PerParticleParameter", name=pname)
    for i in range(natoms):
        ET.SubElement(cnf_elem, "Atom",
                      type=type_ids[i],
                      A=f"{params['eps'][i]:.10e}",
                      B=f"{params['lamb'][i]:.10e}",
                      C6=f"{params['C6'][i]:.10e}",
                      rvdw=f"{params['Rvdw'][i]:.10e}",
                      CTE=f"{params['ct_eps'][i]:.10e}",
                      CTL=f"{params['ct_lamb'][i]:.10e}")

    # ── ByteFF14Force ─────────────────────────────────────────────────
    # Adds back 1-4 (×0.5) and 1-5 (×1.0) VdW interactions excluded above.
    # The charge-transfer (CTE/CTL) term is intentionally omitted for short-range
    # pairs, matching generate_openmm_system / AmoebaCalculator behaviour.
    lj14scale = 0.5
    lj15scale = 1.0
    f14_func = (
        f"S*6*A*exp(B*(1-r/rvdw))-S*C6/rvdw^6/({disp_damping}+(r/rvdw)^6)"
        f"+S*({s12}/r)^12"
    )
    f14_elem = ET.SubElement(root, "ByteFF14Force", energy=f14_func)
    for pname in ["A", "B", "C6", "rvdw", "S"]:
        ET.SubElement(f14_elem, "PerBondParameter", name=pname)
    for scale, pair_key in [(lj14scale, '1-4'), (lj15scale, '1-5')]:
        for i, j in sorted(pairs[pair_key]):
            A_ij  = float(np.sqrt(params['eps'][i]  * params['eps'][j]))
            B_ij  = float(np.sqrt(params['lamb'][i] * params['lamb'][j]))
            C6_ij = float(np.sqrt(params['C6'][i]   * params['C6'][j]))
            r_ij  = 0.5 * (params['Rvdw'][i]        + params['Rvdw'][j])
            ET.SubElement(f14_elem, "Bond",
                          type1=type_ids[i], type2=type_ids[j],
                          A=f"{A_ij:.10e}", B=f"{B_ij:.10e}",
                          C6=f"{C6_ij:.10e}", rvdw=f"{r_ij:.10e}",
                          S=f"{scale:.4f}")

    # ── Write ─────────────────────────────────────────────────────────
    xml_str = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty = '\n'.join(dom.toprettyxml(indent='  ').split('\n')[1:])  # drop <?xml?>
    with open(output_path, 'w') as f:
        f.write(pretty)
    print(f"ForceField XML written to {output_path}  (resname={resname})")


# ── Main workflow ─────────────────────────────────────────────────────────────

def generate_ff_xml_pdb_from_smiles(smiles: str, mol_name: str,
                                     output_dir: str) -> None:
    """Generate ForceField XML and PDB from a SMILES string."""
    params_dir  = os.path.join(output_dir, "params")
    working_dir = os.path.join(output_dir, "working")
    os.makedirs(params_dir,  exist_ok=True)
    os.makedirs(working_dir, exist_ok=True)

    proto = Protocol(params_dir=params_dir, output_dir=output_dir)

    print(f"Generating parameters for {mol_name} ({smiles})...")
    nonbonded_params = proto.generate_ff_params({mol_name: smiles})

    print(f"Building gas-phase system for {mol_name}...")
    proto.build_system(total_atoms=1, components_ratio={mol_name: 1},
                       working_dir=working_dir, build_gas=True)

    top_file = os.path.join(params_dir, "system_gas.top")
    top = GromacsTopFile(top_file)
    bonded_system = top.createSystem(nonbondedMethod=app.NoCutoff,
                                     removeCMMotion=False)

    # ── ForceField XML ────────────────────────────────────────────────
    xml_path = os.path.join(output_dir, f"{mol_name}_ff.xml")
    generate_ff_xml(mol_name, xml_path, top, bonded_system, nonbonded_params)

    # ── PDB with correct resname ──────────────────────────────────────
    from bytemol.core import Molecule
    mol = Molecule.from_smiles(smiles, nconfs=1)
    positions = mol.conformers[0].coords  # Angstrom

    resname = resname_from_mol(mol_name)
    pdb_path = os.path.join(output_dir, f"{mol_name}.pdb")

    buf = io.StringIO()
    app.PDBFile.writeFile(top.topology, positions / 10 * omm.unit.nanometers, buf)
    # Replace the generic 'UNL' residue name with the molecule-specific one
    pdb_text = buf.getvalue().replace(" UNL ", f" {resname} ")
    with open(pdb_path, 'w') as f:
        f.write(pdb_text)
    print(f"PDB written to {pdb_path}  (resname={resname})")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate OpenMM ForceField XML and PDB from SMILES")
    parser.add_argument("--smiles", required=True, help="SMILES string")
    parser.add_argument("--name",   default="molecule", help="Molecule name")
    parser.add_argument("--output", default="output",   help="Output directory")
    args = parser.parse_args()
    generate_ff_xml_pdb_from_smiles(args.smiles, args.name, args.output)
