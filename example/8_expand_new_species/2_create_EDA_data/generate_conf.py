import subprocess
import ase.io
from bytemol.core import Molecule

# --- 修改模版：使用 optimize() 而不是 gradient() ---
PSI4_TEMP = '''
memory 4 GB

molecule {{
{mol_lines}
}}

set {{
  basis def2-svpd
  scf_type df
  reference rks
  geom_maxiter 100  # 设置最大优化步数
}}

# 直接让 Psi4 进行优化，而不是只算梯度
optimize('b3lyp-d3bj')
'''

def gen_molecule_lines(atoms, net_charge):
    # (保持不变)
    spin_multiplicity = 1
    lines = [f"  {net_charge} {spin_multiplicity}\n"]
    for symb, coord in zip(atoms.symbols, atoms.positions):
        line = "  {:3} {:12.8f}  {:12.8f}  {:12.8f}\n".format(symb, coord[0], coord[1], coord[2])
        lines.append(line)
    return lines

def main(mapped_smiles, mol_name):
    # 1. 准备分子 (保持不变)
    mol = Molecule.from_mapped_smiles(mapped_smiles, nconfs=1, name=mol_name)
    net_charge = int(sum(mol.formal_charges))
    atoms = mol.conformers[0].to_ase_atoms()
    
    # 2. 生成 Psi4 输入文件
    mol_lines = gen_molecule_lines(atoms, net_charge)
    psi4_input_content = PSI4_TEMP.format(mol_lines="".join(mol_lines))
    psi4_in = f"./{mol_name}.psi4"
    psi4_out = f"./{mol_name}.out"  # 定义输出文件
    
    with open(psi4_in, "w") as f:
        f.write(psi4_input_content)

    # 3. 直接调用 psi4 命令
    # -n 8 指定核数
    command = f"psi4 -n 8 -i {psi4_in} -o {psi4_out}"
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

    # 4. 解析结果
    # Psi4 运行完会生成一个 .xyz 文件，通常命名为 "output.xyz" 或者 "xyz文件_序号.xyz"
    # 但最稳妥的方法是用 ASE 读取 Psi4 的输出文件 (.out)
    
    try:
        # ASE 可以直接读取 Psi4 的 .out 输出文件中的最终结构
        final_atoms = ase.io.read(psi4_out) 
        
        final_atoms.info['mapped_smiles'] = mapped_smiles
        final_atoms.info['name'] = mol_name
        
        ase.io.write(f"./{mol_name}.xyz", final_atoms)
        print(f"Optimization done via Psi4 native optking.")
        
    except Exception as e:
        print(f"Failed to parse Psi4 output: {e}")

if __name__ == '__main__':
    # (保持不变)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol_name", type=str, default="ACT")
    parser.add_argument("--mapped_smiles", type=str, default="[C:1]([C:2](=[O:3])[C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]")
    args = parser.parse_args()
    main(args.mapped_smiles, args.mol_name)