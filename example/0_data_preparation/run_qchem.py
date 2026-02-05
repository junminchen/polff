import subprocess
import argparse
import ase
import ase.io
import h5py
import os
from ase.units import Bohr
from bytemol.core import Molecule

QCHEM_TEMP = '''$molecule
{mol_lines}
$end

$rem
  JOBTYPE FORCE
  METHOD b3lyp
  BASIS def2-svpd
  DFT_D D3_BJ
  SCF_CONVERGENCE 11
  THRESH 14
  MAX_SCF_CYCLES 50
  SYMMETRY FALSE
  SYM_IGNORE TRUE
  PURECART 1
$end

$archive
enable_archive = True
$end
'''

def gen_molecule_lines(atoms, net_charge):
    spin_multiplicity = 1
    lines = [f"  {net_charge} {spin_multiplicity}"]
    for symb, coord in zip(atoms.symbols, atoms.positions):
        line = "  {:3} {:12.8f}  {:12.8f}  {:12.8f}".format(symb, coord[0], coord[1], coord[2])
        lines.append(line)
    return "\n".join(lines)

def main(mapped_smiles, mol_name, nt):
    mol = Molecule.from_mapped_smiles(mapped_smiles, nconfs=1, name=mol_name)
    net_charge = int(sum(mol.formal_charges))
    atoms = mol.conformers[0].to_ase_atoms()
    mol_lines = gen_molecule_lines(atoms, net_charge)
    
    qc_input_content = QCHEM_TEMP.format(mol_lines=mol_lines)
    qcin = f"./{mol_name}.qcin"
    with open(qcin, "w") as f:
        f.write(qc_input_content)
    
    command = f"geometric-optimize --nt {nt} --converge set GAU --engine qchem {qcin}"
    print(f"Executing: {command}")
    subprocess.run(command, shell=True, check=True)
    
    scratch_env = os.environ.get('QCSCRATCH', './')
    
    possible_paths = [
        os.path.join(scratch_env, "qarchive.h5"), 
        f"./{mol_name}.tmp/run.d/qarchive.h5",   
        "./qarchive.h5"                        
    ]
    
    h5_file = None
    for path in possible_paths:
        if os.path.exists(path):
            h5_file = path
            break
            
    if h5_file is None:
        print(f"Error: Could not find qarchive.h5. Checked: {possible_paths}")
        if os.path.exists(scratch_env):
            print(f"Files in QCSCRATCH ({scratch_env}): {os.listdir(scratch_env)}")
        return

    print(f"Reading data from: {h5_file}")
    try:
        with h5py.File(h5_file, 'r') as f:
            last_job_id = sorted(map(int, list(f['job'].keys())))
            last_job = f['job'][str(last_job_id[-1])]['sp']
            new_atoms = ase.Atoms(
                numbers=last_job['structure']['nuclei'][()],
                positions=last_job['structure']['coordinates'][()] * Bohr
            )
            new_atoms.info['mapped_smiles'] = mapped_smiles
            new_atoms.info['name'] = mol_name
            ase.io.write(f"./{mol_name}.xyz", new_atoms)
            print(f"Successfully saved final structure to {mol_name}.xyz")
    except Exception as e:
        print(f"Error reading H5 file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol_name", type=str, default="ACT")
    parser.add_argument("--mapped_smiles", type=str, default="[C:1]([C:2](=[O:3])[C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]")
    parser.add_argument("--nt", type=int, default=8)
    args = parser.parse_args()
    main(args.mapped_smiles, args.mol_name, args.nt)
