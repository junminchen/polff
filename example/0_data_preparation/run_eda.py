import glob
import json
import re
import subprocess
import os
from bytemol.core import Molecule

# Q-Chem EDA2 Input Template
QCHEM_TEMP = '''{mol_lines}
$rem
  JOBTYPE eda
  EDA2 1
  METHOD WB97M-V
  BASIS def2-tzvpd
  SCF_CONVERGENCE 11
  THRESH 14
  MAX_SCF_CYCLES 100
  SYMMETRY FALSE
  SYM_IGNORE TRUE
  FD_MAT_VEC_PROD false
$end
$archive
enable_archive = True
$end
'''

def gen_dimer_lines(atoms_list, net_charges_list):
    """Generates the $molecule section with fragments separated by '--'"""
    spin_multiplicity = 1
    total_charge = sum(net_charges_list)
    
    lines = ["$molecule\n", f"{total_charge} {spin_multiplicity}\n"]
    
    # Iterate through fragments
    for atoms, charge in zip(atoms_list, net_charges_list):
        lines.append('--\n')
        lines.append(f'{charge} 1\n') # Fragment charge and multiplicity
        for symb, coord in zip(atoms.symbols, atoms.positions):
            line = "  {:3} {:12.8f}  {:12.8f}  {:12.8f}\n".format(symb, coord[0], coord[1], coord[2])
            lines.append(line)
    lines.extend(["$end\n"])
    return lines

def parse_log(log_path):
    """Parses Q-Chem EDA2 log and maps results to English labels"""
    marker = '        Results of EDA2         '
    begin = None
    with open(log_path) as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if marker in line:
                begin = i
                break

    if begin is None:
        raise ValueError(f"Could not find EDA2 results marker in {log_path}")

    # Extract the relevant block
    text = ''.join(lines[begin:begin + 60])
    raw_values = {}

    # Extract patterns: NAME = VALUE or NAME VALUE
    patterns = [
        re.compile(r'(\b[A-Z\s]+)\s*(-?\d+\.\d+)\b'),
        re.compile(r'(\b[A-Z\s]+)=\s*(-?\d+\.\d+)')
    ]
    
    for p in patterns:
        for match in p.findall(text):
            name, val = match[0].strip(), float(match[1])
            if name: raw_values[name] = val

    # Map raw Q-Chem output names to clean English labels
    # Conversion from kJ/mol to kcal/mol (/ 4.184)
    results = {
        "Electrostatics": raw_values.get('ELEC', 0.0) / 4.184,
        "Pauli_Repulsion": raw_values.get('PAULI', 0.0) / 4.184,
        "Dispersion": raw_values.get('DISP', 0.0) / 4.184,
        "Frozen_Energy": raw_values.get('FROZEN', 0.0) / 4.184,
        "Polarization": raw_values.get('POLARIZATION', 0.0) / 4.184,
        "Charge_Transfer": raw_values.get('CHARGE TRANSFER', 0.0) / 4.184,
        "Total_Interaction": raw_values.get('TOTAL', 0.0) / 4.184
    }

    # Verification: Frozen = Elec + Pauli + Disp
    if abs(results["Frozen_Energy"] - (results["Electrostatics"] + results["Pauli_Repulsion"] + results["Dispersion"])) > 0.01:
        print("Warning: EDA components do not sum up to Frozen Energy correctly.")

    return results

def main(input_dir):
    # 1. Identify XYZ files (handling _1, _2 naming)
    xyz_files = sorted(glob.glob(os.path.join(input_dir, '*.xyz')))
    if len(xyz_files) != 2:
        print(f"Error: Found {len(xyz_files)} files in {input_dir}. Exactly 2 required.")
        return

    mols = [Molecule.from_xyz(f) for f in xyz_files]
    atoms_list = []
    charges_list = []
    names_list = []

    for mol in mols:
        atoms_list.append(mol.conformers[0].to_ase_atoms())
        charges_list.append(int(sum(mol.formal_charges)))
        # Clean names like ACT_1 -> ACT
        clean_name = re.sub(r'_\d+$', '', mol.name)
        names_list.append(clean_name)

    # 2. Generate and write Q-Chem Input
    dimer_lines = gen_dimer_lines(atoms_list, charges_list)
    qc_input = QCHEM_TEMP.format(mol_lines="".join(dimer_lines))
    
    job_id = f"{names_list[0]}_{names_list[1]}"
    qcin = f"./{job_id}.qcin"
    qcout = f"./{job_id}.out"

    with open(qcin, 'w') as f:
        f.write(qc_input)

    # 3. Run Q-Chem
    print(f"Starting Q-Chem EDA calculation for {job_id}...")
    # Use local scratch to avoid I/O bottlenecks
    command = f"QCSCRATCH=./ qchem -nt 32 {qcin} {qcout}"
    subprocess.run(command, shell=True, check=True)

    # 4. Parse and Save JSON
    if os.path.exists(qcout):
        try:
            results = parse_log(qcout)
            json_file = f"{job_id}.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Calculation complete. Results saved in English to {json_file}")
        except Exception as e:
            print(f"Error parsing log file: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Directory containing the two fragment XYZ files')
    args = parser.parse_args()
    
    if args.input_dir:
        main(args.input_dir)
    else:
        parser.print_help()
