import json
import os
from pathlib import Path

# ==========================================
# 1. 基础数据库 (Chemical Database)
# ==========================================
# MW: g/mol, Rho: g/cm3, Atoms: total atoms count
chemicals = {
    # Salts (包含离子拆分信息)
    'LiFSI':  {'mw': 187.07, 'rho': 2.30, 'atoms': 10, 'cation': 'LI', 'anion': 'FSI'},
    'LiTFSI': {'mw': 287.09, 'rho': 2.30, 'atoms': 16, 'cation': 'LI', 'anion': 'TFSI'},
    'NaFSI':  {'mw': 203.18, 'rho': 2.40, 'atoms': 10, 'cation': 'NA', 'anion': 'FSI'},
    
    # Solvents
    'DEC':    {'mw': 118.13, 'rho': 0.975, 'atoms': 18},
    'DME':    {'mw': 90.12,  'rho': 0.867, 'atoms': 16},
    'DOL':    {'mw': 74.08,  'rho': 1.060, 'atoms': 11},
    'DMC':    {'mw': 90.08,  'rho': 1.070, 'atoms': 12},
    'EC':     {'mw': 88.06,  'rho': 1.320, 'atoms': 10},
    'PC':     {'mw': 102.09, 'rho': 1.200, 'atoms': 13},
    'G4':     {'mw': 222.28, 'rho': 1.009, 'atoms': 37},
}

# SMILES 映射表
smiles_db = {
    "LI": "[Li+]",
    "NA": "[Na+]",
    "FSI": "O=S(=O)(F)[N-]S(=O)(=O)F",
    "TFSI": "O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F",
    "PF6": "F[P-](F)(F)(F)(F)F",
    "DEC": "CCOC(=O)OCC",
    "DME": "COCCOC",
    "DOL": "C1COCO1",
    "DMC": "COC(=O)OC",
    "EC": "O=C1OCCO1",
    "PC": "CC1COC(=O)O1",
    "G4": "COCCOCCOCCOCCOC"
}

# ==========================================
# 2. 摩尔比配方列表 (Molar Ratio Recipes)
# ==========================================
# 格式: (Name, Salt_Name, Salt_Moles, {Solvent_Name: Moles})
# 这里的数字代表相对摩尔数
recipes_molar = [
    # LiTFSI 变浓度系列 (LiTFSI : DME : DOL = X : 3.57 : 2.40)
    ("LiTFSI_0.1M", "LiTFSI", 0.1, {"DME": 3.57, "DOL": 2.40}),
    ("LiTFSI_0.5M", "LiTFSI", 0.5, {"DME": 3.57, "DOL": 2.40}),
    ("LiTFSI_1.0M", "LiTFSI", 1.0, {"DME": 3.57, "DOL": 2.40}),
    ("LiTFSI_2.0M", "LiTFSI", 2.0, {"DME": 3.57, "DOL": 2.40}),
    ("LiTFSI_3.0M", "LiTFSI", 3.0, {"DME": 3.57, "DOL": 2.40}),
    ("LiTFSI_4.0M", "LiTFSI", 4.0, {"DME": 3.57, "DOL": 2.40}),
]

# ==========================================
# 3. 计算核心逻辑 (Molar Calculation Logic)
# ==========================================
def calculate_composition_molar(target_atoms, salt_name, salt_moles, solvent_moles_dict):
    """
    根据摩尔比例计算分子数，并缩放至目标原子总数。
    """
    salt = chemicals[salt_name]
    
    # 1. 计算当前摩尔比例下的基础总原子数
    total_atoms_unit = salt_moles * salt['atoms']
    for solv_name, moles in solvent_moles_dict.items():
        total_atoms_unit += moles * chemicals[solv_name]['atoms']
    
    # 2. 计算缩放因子 (Scale Factor)
    scale_factor = target_atoms / total_atoms_unit
    
    # 3. 计算并取整分子数
    final_N_salt = int(round(salt_moles * scale_factor))
    final_N_solvents = {k: int(round(v * scale_factor)) for k, v in solvent_moles_dict.items()}
    
    return final_N_salt, final_N_solvents

# ==========================================
# 4. 执行生成 (Execution)
# ==========================================
def main():
    # 设置输出目录
    output_dir = Path("generated_json_molar")
    output_dir.mkdir(exist_ok=True)
    
    target_atoms = 10000 # 目标系统总原子数
    
    print(f"Start processing {len(recipes_molar)} molar ratio recipes...\n")

    for name, salt_key, s_moles, solv_moles_dict in recipes_molar:
        try:
            # 1. 计算缩放后的分子数
            n_salt, n_solvs = calculate_composition_molar(target_atoms, salt_key, s_moles, solv_moles_dict)
            
            # 2. 构建 Components (拆分阳离子/阴离子)
            salt_info = chemicals[salt_key]
            components = {}

            # 先添加溶剂 (保持与 generate_system_gro 的预期顺序一致)
            for s_name, s_count in n_solvs.items():
                components[s_name] = s_count

            # 添加离子
            components[salt_info['cation']] = n_salt
            components[salt_info['anion']] = n_salt
            
            # 3. 构建 Smiles 字典
            used_smiles = {}
            for comp in components.keys():
                if comp in smiles_db:
                    used_smiles[comp] = smiles_db[comp]
                else:
                    print(f"Warning: No SMILES found for {comp}")

            # 4. 组装 JSON 结构
            json_data = {
                "protocol": "Transport",
                "params_dir": "params",
                "output_dir": f"transport_results/{name}",
                "working_dir": f"transport_working_dir/{name}",
                "temperature": 298,
                "natoms": target_atoms,
                "components": components,
                "smiles": used_smiles
            }
            
            # 5. 写入文件
            file_path = output_dir / f"{name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)
                
            print(f"[OK] Generated: {file_path}")
            print(f"    Composition: Salt({salt_key})={n_salt}, Solvents={n_solvs}")

        except Exception as e:
            print(f"[Error] Failed to process {name}: {e}")

    print(f"\nAll done! Files are in the '{output_dir}' folder.")

if __name__ == "__main__":
    main()
