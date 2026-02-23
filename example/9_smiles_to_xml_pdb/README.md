# Generate Force Field XML and PDB from SMILES

This example demonstrates how to automatically generate OpenMM-compatible XML (serialized system) and PDB structure files from a SMILES string using the `byteff2` toolkit.

## Usage

You can run the script `generate_xml_pdb.py` with a SMILES string as input:

```bash
# Example for Ethanol
python generate_xml_pdb.py --smiles "CCO" --name ethanol --output ./ethanol_files
```

The script will produce:
- `ethanol.xml`: A serialized OpenMM `System` object containing all the force field parameters (AmoebaMultipole, CustomNonbonded, etc.).
- `ethanol.pdb`: The molecular structure.
- `params/`: A directory containing intermediate Gromacs-format parameters (`.itp`, `.atp`, `.gro`).

## Workflow Details

1. **Parameter Generation:** Uses `Protocol.generate_ff_params()` which internally uses a trained GNN model to predict atomic-level force field parameters (charge, polarizability, LJ-like parameters).
2. **System Building:** Uses `Protocol.build_system(build_gas=True)` to create a gas-phase Gromacs topology (`.top`) and coordinate (`.gro`) file for the molecule.
3. **OpenMM Integration:** Uses `generate_openmm_system()` to convert the Gromacs topology and the predicted non-bonded parameters into an OpenMM `System` object.
4. **Serialization:** The resulting OpenMM `System` is serialized into an XML file using `openmm.XmlSerializer`.

This workflow provides a convenient way to get a fully parameterized OpenMM system for any molecule starting from its SMILES string.
