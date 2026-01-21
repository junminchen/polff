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
    "PF6": "F[P-](F)(F)(F)(F)F", # 预留
    "DEC": "CCOC(=O)OCC",
    "DME": "COCCOC",
    "DOL": "C1COCO1",
    "DMC": "COC(=O)OC",
    "EC": "O=C1OCCO1",
    "PC": "CC1COC(=O)O1",
    "G4": "COCCOCCOCCOCCOC"
}

# ==========================================
# 2. 配方列表 (Recipes)
# ==========================================
# (Name, Salt, Molarity, {Solvent: Ratio})
recipes = [
    # Core References
    # ("ELref",       "LiFSI",  1.0, {"DEC": 1}),
    # ("Salt_Bridge", "LiTFSI", 3.0, {"DME": 1, "DOL": 1}),
    # ("ELtest",      "LiFSI",  1.0, {"DME": 1}),
    
    # Li Comparisons
    ("Li_DMC",      "LiFSI",  1.0, {"DMC": 1}),
    ("Li_G4",       "LiFSI",  1.0, {"G4": 1}),
    ("Li_PC",       "LiFSI",  1.0, {"PC": 1}),
    ("Li_EC",       "LiFSI",  1.0, {"EC": 1}),
    ("Li_DEC",      "LiFSI",  1.0, {"DEC": 1}),
    
    # Na Comparisons
    ("Na_DMC",      "NaFSI",  1.0, {"DMC": 1}),
    ("Na_G4",       "NaFSI",  1.0, {"G4": 1}),
    ("Na_PC",       "NaFSI",  1.0, {"PC": 1}),
    ("Na_EC",       "NaFSI",  1.0, {"EC": 1}),
    ("Na_DEC",      "NaFSI",  1.0, {"DEC": 1}),
]

# ==========================================
# 3. 计算核心逻辑 (Calculation Logic)
# ==========================================
def calculate_composition(target_atoms, salt_name, molarity, solvent_ratios):
    """
    Input: 目标原子数, 盐名称, 摩尔浓度(mol/L), 溶剂体积比字典
    Output: 缩放后的盐分子数(int), 缩放后的溶剂分子数字典({name: int})
    """
    salt = chemicals[salt_name]
    
    # Step A: 1L 基准计算
    # 1. 盐的量
    n_salt_base = molarity 
    vol_salt = (n_salt_base * salt['mw']) / salt['rho']
    
    # 2. 溶剂总可用体积
    vol_solvents_total = 1000.0 - vol_salt
    if vol_solvents_total <= 0:
        raise ValueError(f"Salt volume ({vol_salt:.2f} mL) exceeds total volume for {salt_name}!")

    # 3. 各溶剂摩尔数
    total_ratio = sum(solvent_ratios.values())
    solvents_moles_base = {}
    
    for solv_name, ratio in solvent_ratios.items():
        solv = chemicals[solv_name]
        vol_this = vol_solvents_total * (ratio / total_ratio)
        n_this = (vol_this * solv['rho']) / solv['mw']
        solvents_moles_base[solv_name] = n_this

    # Step B: 缩放至目标原子数 (Target Atoms Scaling)
    # 1. 计算基准总原子数
    total_atoms_base = (n_salt_base * salt['atoms'])
    for name, n in solvents_moles_base.items():
        total_atoms_base += n * chemicals[name]['atoms']
        
    # 2. 缩放因子
    scale_factor = target_atoms / total_atoms_base
    
    # 3. 应用缩放并取整
    final_N_salt = int(round(n_salt_base * scale_factor))
    final_N_solvents = {k: int(round(v * scale_factor)) for k, v in solvents_moles_base.items()}
    
    return final_N_salt, final_N_solvents

# ==========================================
# 4. 执行生成 (Execution)
# ==========================================
def main():
    # 设置输出目录
    output_dir = Path("generated_json")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Start processing {len(recipes)} recipes...\n")

    for name, salt_key, m, solv_dict in recipes:
        try:
            # 1. 计算分子数
            n_salt, n_solvs = calculate_composition(10000, salt_key, m, solv_dict)
            
            # 2. 构建 Components (拆分阳离子/阴离子)
            salt_info = chemicals[salt_key]
            components = {}
            
            # Add Ions
            components[salt_info['cation']] = n_salt
            components[salt_info['anion']] = n_salt
            
            # Add Solvents
            for s_name, s_count in n_solvs.items():
                components[s_name] = s_count
            
            # 3. 构建 Smiles 字典 (仅包含用到的组分)
            used_smiles = {}
            for comp in components.keys():
                if comp in smiles_db:
                    used_smiles[comp] = smiles_db[comp]
                else:
                    print(f"Warning: No SMILES found for {comp}")

            # 4. 组装最终 JSON 结构
            json_data = {
                "protocol": "Transport",
                "params_dir": "params",
                "output_dir": f"transport_results/{name}",
                "working_dir": f"transport_working_dir/{name}",
                "temperature": 298,
                "natoms": 10000,
                "components": components,
                "smiles": used_smiles
            }
            
            # 5. 写入文件
            file_path = output_dir / f"{name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)
                
            print(f"[OK] Generated: {file_path} (Salt: {n_salt}, Solvents: {n_solvs})")

        except Exception as e:
            print(f"[Error] Failed to process {name}: {e}")

    print("\nAll done! Files are in the 'generated_json' folder.")

if __name__ == "__main__":
    main()