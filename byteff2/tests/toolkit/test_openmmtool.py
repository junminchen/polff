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

import json
import os
import tempfile

import openmm as omm
import openmm.app as app
import openmm.unit as openmm_unit
import pytest

from byteff2.toolkit.openmmtool import load_openmm_system
from byteff2.toolkit.protocol import Component, ComponentType


def create_simple_pdb_file(pdb_path):
    """Create a simple PDB file for testing."""
    pdb_content = """CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1  O   WAT     1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H1  WAT     1       0.757   0.586   0.000  1.00  0.00           H
ATOM      3  H2  WAT     1      -0.757   0.586   0.000  1.00  0.00           H
END
"""
    with open(pdb_path, 'w') as f:
        f.write(pdb_content)


def create_simple_system():
    """Create a simple OpenMM System for testing."""
    system = omm.System()
    
    # Add three particles (for a water molecule)
    system.addParticle(16.0 * openmm_unit.dalton)  # Oxygen
    system.addParticle(1.0 * openmm_unit.dalton)   # Hydrogen 1
    system.addParticle(1.0 * openmm_unit.dalton)   # Hydrogen 2
    
    # Add a simple harmonic bond force
    bond_force = omm.HarmonicBondForce()
    bond_force.addBond(0, 1, 0.1 * openmm_unit.nanometer, 1000.0 * openmm_unit.kilojoule_per_mole / openmm_unit.nanometer**2)
    bond_force.addBond(0, 2, 0.1 * openmm_unit.nanometer, 1000.0 * openmm_unit.kilojoule_per_mole / openmm_unit.nanometer**2)
    system.addForce(bond_force)
    
    # Add an AmoebaMultipoleForce with charges for testing
    amoeba_force = omm.AmoebaMultipoleForce()
    
    # Add multipoles for water (O, H, H) with typical charges
    dipoles = [0.0, 0.0, 0.0]
    quadrupoles = [0.0] * 9
    
    # Oxygen with negative charge
    amoeba_force.addMultipole(-0.8, dipoles, quadrupoles,
                             omm.AmoebaMultipoleForce.NoAxisType, 0, 1, 2,
                             0.39, 1.0, 0.1)
    
    # Hydrogen 1 with positive charge
    amoeba_force.addMultipole(0.4, dipoles, quadrupoles,
                             omm.AmoebaMultipoleForce.NoAxisType, 0, 1, 2,
                             0.39, 1.0, 0.1)
    
    # Hydrogen 2 with positive charge
    amoeba_force.addMultipole(0.4, dipoles, quadrupoles,
                             omm.AmoebaMultipoleForce.NoAxisType, 0, 1, 2,
                             0.39, 1.0, 0.1)
    
    amoeba_force.setNonbondedMethod(omm.AmoebaMultipoleForce.NoCutoff)
    system.addForce(amoeba_force)
    
    return system


def test_load_openmm_system():
    """Test loading OpenMM system from PDB and XML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, 'test.pdb')
        xml_path = os.path.join(tmpdir, 'test.xml')
        
        # Create test files
        create_simple_pdb_file(pdb_path)
        system = create_simple_system()
        
        # Serialize system to XML
        with open(xml_path, 'w') as f:
            f.write(omm.XmlSerializer.serialize(system))
        
        # Test loading
        pdb, loaded_system = load_openmm_system(pdb_path, xml_path)
        
        # Verify results
        assert isinstance(pdb, app.PDBFile)
        assert isinstance(loaded_system, omm.System)
        assert loaded_system.getNumParticles() == 3
        assert loaded_system.getNumForces() == 2


def test_component_traditional_init():
    """Test Component initialization with topo_mol (traditional method)."""
    # Create a mock topo_mol object
    class MockAtom:
        def __init__(self, charge, mass):
            self.charge = charge
            self.mass = mass
    
    class MockTopoMol:
        def __init__(self, name, atoms):
            self.name = name
            self.atoms = atoms
    
    # Test neutral molecule (solvent)
    atoms = [MockAtom(0.0, 16.0), MockAtom(0.0, 1.0), MockAtom(0.0, 1.0)]
    topo_mol = MockTopoMol('WAT', atoms)
    component = Component(topo_mol=topo_mol)
    
    assert component.name == 'WAT'
    assert component.type == ComponentType.SOLVENT
    assert component.net_charge == 0.0
    assert component.molar_mass == 18.0
    assert component.density == 0.9


def test_component_manual_init():
    """Test Component initialization without topo_mol (manual method)."""
    # Test cation
    component = Component(name='LI', charge=1.0, mass=6.94)
    assert component.name == 'LI'
    assert component.type == ComponentType.CATION
    assert component.net_charge == 1.0
    assert component.molar_mass == 6.94
    assert component.density == 0.25
    assert component.atoms == []
    
    # Test anion
    component = Component(name='CL', charge=-1.0, mass=35.45)
    assert component.name == 'CL'
    assert component.type == ComponentType.ANION
    assert component.net_charge == -1.0
    
    # Test neutral molecule
    component = Component(name='H2O', charge=0.0, mass=18.0)
    assert component.name == 'H2O'
    assert component.type == ComponentType.SOLVENT


def test_component_manual_init_validation():
    """Test that manual Component initialization validates input."""
    # Test that it raises ValueError when missing required parameters
    with pytest.raises(ValueError):
        Component(name='TEST')  # Missing charge and mass
    
    with pytest.raises(ValueError):
        Component(charge=0.0, mass=10.0)  # Missing name
    
    with pytest.raises(ValueError):
        Component(name='TEST', charge=0.0)  # Missing mass


def test_reconstruct_components():
    """Test reconstructing components from PDB and System."""
    from byteff2.toolkit.protocol import Protocol
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, 'test.pdb')
        xml_path = os.path.join(tmpdir, 'test.xml')
        
        # Create test files
        create_simple_pdb_file(pdb_path)
        system = create_simple_system()
        
        with open(xml_path, 'w') as f:
            f.write(omm.XmlSerializer.serialize(system))
        
        # Load system
        pdb, loaded_system = load_openmm_system(pdb_path, xml_path)
        
        # Create Protocol instance
        protocol = Protocol(params_dir=tmpdir, output_dir=tmpdir)
        
        # Reconstruct components
        components = protocol.reconstruct_components(pdb, loaded_system)
        
        # Verify results
        assert isinstance(components, dict)
        assert 'WAT' in components
        assert components['WAT'].name == 'WAT'
        assert components['WAT'].molar_num == 1
        # Charge should be close to neutral (sum of -0.8 + 0.4 + 0.4 = 0.0)
        assert abs(components['WAT'].net_charge) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
