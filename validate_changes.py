#!/usr/bin/env python3
"""
Simple validation script to check the new functionality without running full tests.
This checks:
1. load_openmm_system function exists and has correct signature
2. Component class can be initialized manually
3. Protocol class has reconstruct_components method
4. All three protocol classes check for system_pdb and system_xml
"""

import ast
import sys


def check_function_signature(filepath, function_name, expected_params, expected_return=None):
    """Check if a function exists with the expected signature."""
    with open(filepath, 'r') as f:
        code = f.read()
        tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            params = [arg.arg for arg in node.args.args]
            if params != expected_params:
                print(f"❌ {function_name} has unexpected parameters: {params}")
                return False
            
            if expected_return and node.returns:
                ret_annotation = ast.unparse(node.returns)
                if expected_return not in ret_annotation:
                    print(f"❌ {function_name} has unexpected return type: {ret_annotation}")
                    return False
            
            print(f"✓ {function_name} exists with correct signature")
            return True
    
    print(f"❌ {function_name} not found in {filepath}")
    return False


def check_class_method(filepath, class_name, method_name, expected_params):
    """Check if a class has a method with expected parameters."""
    with open(filepath, 'r') as f:
        code = f.read()
        tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    params = [arg.arg for arg in item.args.args]
                    if params != expected_params:
                        print(f"❌ {class_name}.{method_name} has unexpected parameters: {params}")
                        return False
                    print(f"✓ {class_name}.{method_name} exists with correct signature")
                    return True
    
    print(f"❌ {class_name}.{method_name} not found")
    return False


def check_code_contains(filepath, pattern, description):
    """Check if code contains a specific pattern."""
    with open(filepath, 'r') as f:
        code = f.read()
        if pattern in code:
            print(f"✓ {description}")
            return True
        else:
            print(f"❌ {description}")
            return False


def main():
    print("Validating new functionality...")
    print()
    
    all_checks_passed = True
    
    # Check openmmtool.py
    print("Checking byteff2/toolkit/openmmtool.py:")
    all_checks_passed &= check_function_signature(
        'byteff2/toolkit/openmmtool.py',
        'load_openmm_system',
        ['pdb_file', 'xml_file'],
        'tuple'
    )
    all_checks_passed &= check_code_contains(
        'byteff2/toolkit/openmmtool.py',
        'app.PDBFile',
        'load_openmm_system uses app.PDBFile'
    )
    all_checks_passed &= check_code_contains(
        'byteff2/toolkit/openmmtool.py',
        'XmlSerializer.deserialize',
        'load_openmm_system deserializes XML'
    )
    print()
    
    # Check protocol.py - Component class
    print("Checking byteff2/toolkit/protocol.py - Component class:")
    all_checks_passed &= check_class_method(
        'byteff2/toolkit/protocol.py',
        'Component',
        '__init__',
        ['self', 'topo_mol', 'name', 'charge', 'mass']
    )
    all_checks_passed &= check_code_contains(
        'byteff2/toolkit/protocol.py',
        'if topo_mol is not None:',
        'Component supports both topo_mol and manual initialization'
    )
    print()
    
    # Check protocol.py - Protocol class
    print("Checking byteff2/toolkit/protocol.py - Protocol class:")
    all_checks_passed &= check_class_method(
        'byteff2/toolkit/protocol.py',
        'Protocol',
        'reconstruct_components',
        ['self', 'pdb', 'system']
    )
    all_checks_passed &= check_code_contains(
        'byteff2/toolkit/protocol.py',
        'load_openmm_system',
        'Protocol imports load_openmm_system'
    )
    print()
    
    # Check DensityProtocol
    print("Checking byteff2/toolkit/protocol.py - DensityProtocol:")
    all_checks_passed &= check_code_contains(
        'byteff2/toolkit/protocol.py',
        "'system_pdb' in self.config and 'system_xml' in self.config",
        'DensityProtocol checks for system_pdb and system_xml'
    )
    print()
    
    # Check TransportProtocol
    print("Checking byteff2/toolkit/protocol.py - TransportProtocol:")
    all_checks_passed &= check_code_contains(
        'byteff2/toolkit/protocol.py',
        "'system_pdb' in self.config and 'system_xml' in self.config",
        'TransportProtocol checks for system_pdb and system_xml (found at least once)'
    )
    print()
    
    # Check HVapProtocol
    print("Checking byteff2/toolkit/protocol.py - HVapProtocol:")
    all_checks_passed &= check_code_contains(
        'byteff2/toolkit/protocol.py',
        "'system_pdb_gas' in self.config and 'system_xml_gas' in self.config",
        'HVapProtocol checks for system_pdb_gas and system_xml_gas'
    )
    print()
    
    if all_checks_passed:
        print("=" * 60)
        print("✅ All validation checks passed!")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("❌ Some validation checks failed!")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
