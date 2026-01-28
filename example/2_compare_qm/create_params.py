import json
import os
from math import isclose

import ase.io as aio
import numpy as np
import traceback  # 可选：用于打印详细错误栈，方便排查问题

from byteff2.train.utils import get_nb_params, load_model
from byteff2.utils.mol_inventory import all_name_mapped_smiles
from bytemol.core import Molecule
from bytemol.utils import get_data_file_path
# from CreateFormula import five_mapped_smiles, DENSITY_DICT, tenary_mapped_smiles

from mol_bank import *

OUTPUT_DIR = os.path.abspath("./params_results")
REF_DIR = os.path.abspath("./AFGBL")


def compare_floats(a, b, path="", *, rtol=1e-5, atol=1e-8):
    """Recursively compare floats (or nested lists) in two JSON-like objects."""
    if isinstance(a, float) and isinstance(b, float):
        if not isclose(a, b, rel_tol=rtol, abs_tol=atol):
            print(f"MISMATCH at {path}: {a} vs {b}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            print(f"LENGTH MISMATCH at {path}: {len(a)} vs {len(b)}")
            return
        for idx, (ai, bi) in enumerate(zip(a, b)):
            compare_floats(ai, bi, f"{path}[{idx}]", rtol=rtol, atol=atol)
    elif isinstance(a, dict) and isinstance(b, dict):
        for key in a.keys() | b.keys():
            if key not in a or key not in b:
                print(f"KEY MISSING at {path}: {key}")
                continue
            compare_floats(a[key], b[key], f"{path}.{key}", rtol=rtol, atol=atol)
    else:
        # Non-float scalars (int, str, bool) must be exactly equal
        if a != b:
            print(f"NON-FLOAT MISMATCH at {path}: {a} vs {b}")


def write_gro(mol: Molecule, save_path: str):
    atoms_gro = mol.conformers[0].to_ase_atoms()
    atoms_gro.set_array('residuenames', np.array([mol.name] * mol.natoms))
    aio.write(save_path, atoms_gro)


def main():
    # load model
    model_dir = get_data_file_path('optimal.pt', 'byteff2.trained_models')
    model = load_model(os.path.dirname(model_dir))
    # generate input mol
    mols = phyneo_name_mapped_smiles
    
    # for name in DENSITY_DICT.keys():
    for name in mols.keys():
        if name not in ['PF6', 'BOB', 'DFOB']:
            print(name)
            mps = mols[name]
            # mps = mols['FSI']
            OUTPUT_DIR = os.path.abspath(f"./params/{name}")
            if os.path.exists(OUTPUT_DIR):
                print('exists, skip')
            else:
                try:  # 新增：包裹可能出错的mol初始化+get_nb_params逻辑
                    mol = Molecule.from_mapped_smiles(mps, nconfs=1)
                    mol.name = name
                    # generate force field params
                    print(mps)
                    if 'B' not in mps and 'Si' not in mps:
                        # 核心：捕获这行的所有异常
                        metadata, params, tfs, mol = get_nb_params(model, mol)
                        # clean old data
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        if os.path.exists(f'{OUTPUT_DIR}/{mol.name}.json'):
                            os.remove(f'{OUTPUT_DIR}/{mol.name}.json')
                        tfs.write_itp(f'{OUTPUT_DIR}/{mol.name}.itp', separated_atp=True)
                        write_gro(mol, f'{OUTPUT_DIR}/{mol.name}.gro')
                        with open(f'{OUTPUT_DIR}/{mol.name}.json', 'w') as f:
                            json.dump(params, f, indent=2)
                        with open(f'{OUTPUT_DIR}/{mol.name}_nb_params.json', 'w') as file:
                            nb_params = {'metadata': metadata}
                            json.dump(nb_params, file, indent=2)
                except Exception as e:  # 捕获所有异常（也可指定具体异常如AssertionError）
                    # 打印错误信息，便于排查（可选但推荐）
                    print(f"❌ 处理分子 {name} 时出错（get_nb_params调用失败）：{str(e)}")
                    # 可选：打印详细错误栈，定位具体报错行
                    traceback.print_exc()
                    # 跳过当前循环，处理下一个name
                    continue
if __name__ == '__main__':
    main()
