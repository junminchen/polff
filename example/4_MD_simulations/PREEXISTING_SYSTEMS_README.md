# Loading Pre-existing OpenMM Systems

This directory contains example configuration files showing how to use pre-existing OpenMM systems instead of building them from scratch.

## New Functionality

The toolkit now supports loading OpenMM systems from PDB and XML files, bypassing the system building process. This is useful when you:
- Have already created and serialized an OpenMM system
- Want to run MD simulations or post-processing on existing systems
- Need to restart or continue simulations with saved systems

## Usage

### For Density and Transport Protocols

Add the following keys to your configuration file:
- `system_pdb`: Path to the PDB file containing the system structure
- `system_xml`: Path to the serialized OpenMM System XML file

Example (`density_config_preexisting.json`):
```json
{
    "protocol": "Density",
    "params_dir": "density_params",
    "output_dir": "density_results",
    "temperature": 298,
    "system_pdb": "/path/to/your/system.pdb",
    "system_xml": "/path/to/your/system.xml"
}
```

When these keys are present, the protocol will:
1. Load the system from the PDB and XML files
2. Reconstruct component information from the loaded system
3. Skip the force field parameter generation and system building steps
4. Proceed directly to running the MD simulation

### For HVap Protocol

For heat of vaporization calculations, you need both liquid and gas phase systems:
- `system_pdb`: Path to the liquid phase PDB file
- `system_xml`: Path to the liquid phase XML file
- `system_pdb_gas`: Path to the gas phase PDB file
- `system_xml_gas`: Path to the gas phase XML file

Example (`hvap_config_preexisting.json`):
```json
{
    "protocol": "HVap",
    "params_dir": "hvap_params",
    "output_dir": "hvap_results",
    "temperature": 298,
    "system_pdb": "/path/to/your/liquid_system.pdb",
    "system_xml": "/path/to/your/liquid_system.xml",
    "system_pdb_gas": "/path/to/your/gas_system.pdb",
    "system_xml_gas": "/path/to/your/gas_system.xml"
}
```

## Creating System Files

To create PDB and XML files from an existing OpenMM system:

### PDB File
```python
import openmm.app as app

# Assuming you have a topology and positions
with open('system.pdb', 'w') as f:
    app.PDBFile.writeFile(topology, positions, f)
```

### XML File
```python
import openmm as omm

# Assuming you have a system object
with open('system.xml', 'w') as f:
    f.write(omm.XmlSerializer.serialize(system))
```

Or using the AmoebaCalculator:
```python
from byteff2.toolkit.openmmtool import AmoebaCalculator

calculator = AmoebaCalculator(...)
calculator.serialize('system.xml')
```

## Backward Compatibility

The traditional workflow using SMILES and component specifications remains fully supported. If `system_pdb` and `system_xml` are not provided in the configuration, the protocols will automatically use the traditional workflow to generate force field parameters and build systems.

## API Reference

### New Functions

#### `load_openmm_system(pdb_file: str, xml_file: str) -> tuple[app.PDBFile, omm.System]`
Location: `byteff2.toolkit.openmmtool`

Loads a PDB file and a serialized OpenMM System XML file.

**Parameters:**
- `pdb_file`: Path to the PDB file
- `xml_file`: Path to the serialized OpenMM System XML file

**Returns:**
- A tuple of (PDBFile, System)

### Modified Classes

#### `Component`
Location: `byteff2.toolkit.protocol`

The `Component` class now supports manual initialization without `topo_mol`:

```python
# Traditional initialization
component = Component(topo_mol=topo_mol)

# Manual initialization
component = Component(name='LI', charge=1.0, mass=6.94)
```

#### `Protocol.reconstruct_components(pdb: app.PDBFile, system: omm.System) -> dict`
Location: `byteff2.toolkit.protocol`

Reconstructs component information from a loaded PDB and System.

**Parameters:**
- `pdb`: OpenMM PDBFile object
- `system`: OpenMM System object

**Returns:**
- Dictionary of Component objects keyed by component name

## Notes

- When using pre-existing systems, ensure that the PDB and XML files are compatible and represent the same system
- The component reconstruction process infers information from the system, so some metadata may differ from the original component specifications
- The reconstructed components will be used for post-processing steps that require component information (e.g., conductivity calculations in TransportProtocol)
