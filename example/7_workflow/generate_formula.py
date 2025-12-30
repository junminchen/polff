# Electrolyte Bulk MD Simulation SOP
import os
import re
import glob
import shutil
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mol_inventory import * 
import pandas as pd

# ========== 基础数据 ==========
SMILES_DICT = all_name_mapped_smiles

element_to_mass = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
    'P': 30.974, 'S': 32.06, 'CL': 35.45, 'BR': 79.904, 'I': 126.90,
    'B': 10.81, 'SI': 28.085,
    'LI': 6.94, 'NA': 22.990, 'K': 39.098, 'MG': 24.305, 'CA': 40.078, 'AL': 26.982,
}

# ========== 构建数据库 ==========
def smiles_to_molar_mass(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Descriptors.ExactMolWt(mol)

MOLECULE_DB = {}
for name, smiles in SMILES_DICT.items():
    try:
        molar_mass = smiles_to_molar_mass(smiles)
    except Exception as e:
        print(f"⚠️ Error for {name}: {e}")
        continue
    MOLECULE_DB[name] = (round(molar_mass, 2))

mol_charges = {}
for name, smiles in SMILES_DICT.items():
    mol = Chem.MolFromSmiles(smiles)
    total_charge = 0 if mol is None else sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    mol_charges[name] = total_charge

# ========== 分子数计算 ==========
NA = 6.02214076e23
BOX_NM = 5.0
VOL_CM3 = (BOX_NM ** 3) * 1e-21  # nm³ → cm³
VOL_L = VOL_CM3 * 1e-3           # cm³ → L

def mass_fraction_to_counts(mass_fraction_dict, db, solute_to_ions=None, box_nm=5.0):
    NA = 6.02214076e23
    vol_cm3 = (box_nm ** 3) * 1e-21  # nm³ → cm³

    if solute_to_ions is None:
        solute_to_ions = {
            'LiPF6': ('LI', 'PF6'),
            'LiFSI': ('LI', 'FSI'),
            'LiTFSI': ('LI', 'TFSI'),
            'LiDFOB': ('LI', 'DFOB'),
            'NaPF6': ('NA', 'PF6'),
            'NaFSI': ('NA', 'FSI'),
            'NaTFSI': ('NA', 'TFSI'),
        }

    # auto_generate_solute_to_ions ?

    # 分离溶剂和盐类
    solvent_names = [name for name in mass_fraction_dict if name not in solute_to_ions]
    salt_names = [name for name in mass_fraction_dict if name in solute_to_ions]

    # 检查溶剂密度是否缺失
    missing = [name for name in solvent_names if name not in db or db[name][1] is None]
    if missing:
        raise ValueError(f"Missing density info for solvents: {missing}")

    # Normalize fractions
    total = sum(mass_fraction_dict.values())
    weight_fractions = {name: w / total for name, w in mass_fraction_dict.items()}

    # Compute mixture density using only solvents
    inv_rho_mix = sum(weight_fractions[name] / db[name][1] for name in solvent_names)
    rho_mix = 1.0 / inv_rho_mix  # g/cm³

    # Total mass in box
    total_mass_g = rho_mix * vol_cm3

    # Compute counts
    counts = {}

    # Step 1: 溶剂
    for name in solvent_names:
        wf = weight_fractions[name]
        mass_g = total_mass_g * wf
        mol_mass = db[name][0]
        mol_count = int(round((mass_g / mol_mass) * NA))
        counts[name] = mol_count

    # Step 2: 盐类（拆分为离子）
    total_solvent_mass = sum((counts[n] * db[n][0]) / NA for n in solvent_names)
    for name in salt_names:
        wf = weight_fractions[name]
        salt_mass_g = total_solvent_mass * wf / sum(weight_fractions[n] for n in solvent_names)  # scale to solvent mass
        cation, anion = solute_to_ions[name]
        if cation not in db or anion not in db:
            raise KeyError(f"Missing molar mass for ion components of {name}")
        mol_mass = db[cation][0] + db[anion][0]
        mol_count = int(round((salt_mass_g / mol_mass) * NA))
        counts[cation] = counts.get(cation, 0) + mol_count
        counts[anion] = counts.get(anion, 0) + mol_count

    return counts

# ========== PDB 质量提取 ==========
def get_mol_masses():
    def parse_pdb_for_masses(pdb_file):
        masses = []
        with open(pdb_file,'r') as f:
            for line in f:
                if line.startswith(('ATOM','HETATM')):
                    element = line[76:78].strip().upper()
                    if not element:
                        atom_name = line[12:16].strip()
                        elem = ''.join([c for c in atom_name if c.isalpha()])
                        element = elem[:2].upper() if len(elem)>=2 else elem.upper()
                    if element in element_to_mass:
                        masses.append(element_to_mass[element])
        return masses
    mol_masses = {}
    for pdb_file in glob.glob('template/input_data/*/*.pdb'):
        mol_name = os.path.splitext(os.path.basename(pdb_file))[0]
        masses = parse_pdb_for_masses(pdb_file)
        if masses: mol_masses[mol_name] = masses
    return mol_masses

# ========== 文件生成 ==========
def save_run_file(folder, mol_masses, mol_counts, temperature=298,template_dir='/AI4S/Projects/CAFF/ELF_MD_SOP/template'):
    template_py=f"{template_dir}/run_transport.py"
    # template_py = "template/run_transport.py"
    output_py = os.path.join(folder,"run_md.py")
    with open(template_py,'r') as f: content = f.read()
    content = re.sub(r'TEMPERATURE\s*=\s*\d+', f'TEMPERATURE = {temperature}', content)

    number_dict_str = ", ".join([f'"{k}": {v}' for k,v in mol_counts.items()])
    content = re.sub(r'species_number_dict = \{.*?\}', f'species_number_dict = {{{number_dict_str}}}', content, flags=re.DOTALL)
    charge_dict_str = ", ".join([f'"{k}": {mol_charges[k]}' for k in mol_counts])
    content = re.sub(r'species_charges_dict = \{.*?\}', f'species_charges_dict = {{{charge_dict_str}}}', content, flags=re.DOTALL)
    mass_dict_lines = [f'    "{mol}": [{", ".join(str(m) for m in mol_masses.get(mol,[12.0]))}]' for mol in mol_counts]
    mass_dict_full = "species_mass_dict = {\n" + ",\n".join(mass_dict_lines) + "\n}"
    content = re.sub(r'species_mass_dict = \{.*?\}', mass_dict_full, content, flags=re.DOTALL)
    mol_list_str = ", ".join([f"'{k}'" for k in mol_counts])
    content = re.sub(r"for mol in \[.*?\]:", f"for mol in [{mol_list_str}]:", content)
    with open(output_py,'w') as f: f.write(content)

def save_pack_file(folder, mol_counts,template_dir='/AI4S/Projects/CAFF/ELF_MD_SOP/template'):
    template_pack = f"{template_dir}/input_data/pack_data.py"
    output_pack = os.path.join(folder,"prepare_system.py")
    with open(template_pack,'r') as f: pack_content = f.read()
    mol_entries = [f"    '{template_dir}/input_data/{mol}/{mol}.gro': {{'resname': '{mol}', 'count': {count}}}," for mol,count in mol_counts.items()]
    mol_dict_str = "{\n" + "\n".join(mol_entries) + "\n}"
    pack_content = re.sub(r"molecules = \{.*?\}", f"molecules = {mol_dict_str}", pack_content, flags=re.DOTALL)
    total_mols = sum(mol_counts.values())
    # box_size = (total_mols*120)**(1/3)
    # box_size = max(50.0,min(100.0,box_size))
    box_size = 150.0
    pack_content = re.sub(r"box_size = .*?#", f"box_size = {box_size:.1f}  #", pack_content)
    with open(output_pack,'w') as f: f.write(pack_content)

def save_topo_file(folder, mol_counts,template_dir='/AI4S/Projects/CAFF/ELF_MD_SOP/template'):
    topol_content = "[ defaults ]\n1 3 yes 0.5 0.5\n\n[ atomtypes ]\n"
    for mol in mol_counts:
        topol_content += f'#include "{template_dir}/input_data/{mol}/{mol}.atp"\n'
    for mol in mol_counts:
        topol_content += f'#include "{template_dir}/input_data/{mol}/{mol}.itp"\n'
    topol_content += "\n[ system ]\n" + " ".join(mol_counts.keys()) + "\n\n[molecules]\n"
    for mol,count in mol_counts.items():
        topol_content += f"{mol} {count}\n"
    with open(os.path.join(folder,"topol.top"),'w') as f: f.write(topol_content)




def mass_fraction_to_counts(mass_fraction_dict, total_molecules=10000):
    """
    基于总分子数计算各组分数量（完全移除密度依赖）
    核心逻辑：质量分数 → 摩尔分数 → 总分子数按比例分配
    
    参数：
    ----------
    mass_fraction_dict : dict
        质量分数字典（如 {'EC':0.8, 'LiPF6':0.2}）
    total_molecules : int, optional
        体系总分子/离子数（锚定值），默认10000个，可根据需求调整
    
    返回：
    ----------
    dict
        各组分（溶剂/离子）的分子数（整数）
    """
    NA = 6.02214076e23  # 阿伏伽德罗常数（仅用于逻辑对齐，无实际计算）


    # 1. 分离溶剂/盐类
    solvents = [component for component in mass_fraction_dict if component['component_type'] != 'salt']
    salts = [component for component in mass_fraction_dict if component['component_type'] == 'salt']

    # 2. 归一化质量分数（确保总和为1）
    total_mass = sum([component['mass_fraction'] for component in mass_fraction_dict])
    weight_fractions = {component['name']: component['mass_fraction'] / total_mass for component in mass_fraction_dict}

    # 3. 核心：质量分数 → 摩尔当量（摩尔数=质量/摩尔质量）
    mole_equivalent = {}  # 各组分的摩尔当量（无单位，仅用于计算比例）
    total_mole_equivalent = 0.0

    # 3.1 溶剂的摩尔当量
    for solvent in solvents:
        name = solvent['name']
        mol_mass = solvent['molar_mass']
        mole_equivalent[name] = weight_fractions[name] / mol_mass
        total_mole_equivalent += mole_equivalent[name]

    # 3.2 盐的摩尔当量（按阴阳离子总摩尔质量算）
    for salt in salts:
        name = salt['name']
        salt_mol_mass = salt['molar_mass']
        mole_equivalent[name] = weight_fractions[name] / salt_mol_mass
        total_mole_equivalent += mole_equivalent[name]

    # 4. 摩尔分数 = 组分摩尔当量 / 总摩尔当量
    mole_fractions = {
        component['name']: mole_equivalent[component['name']] / total_mole_equivalent 
        for component in solvents + salts
    }

    # 5. 分配总分子数（核心：总分子数 × 摩尔分数）
    counts = {}

    # 5.1 溶剂分子数
    for solvent in solvents:
        name = solvent['name']
        counts[name] = int(round(total_molecules * mole_fractions[name]))

    # 5.2 盐拆分为阴阳离子（1:1比例，分子数与盐的摩尔分数一致）
    for salt in salts:
        name = salt['name']
        salt_mol_count = int(round(total_molecules * mole_fractions[name]))
        cation, anion = salt['cation_name'], salt['anion_name']
        cation, anion = solute_to_ions[name]
        # 阴阳离子各加对应数量（盐拆分为1个阳离子+1个阴离子）
        counts[cation] = counts.get(cation, 0) + salt_mol_count
        counts[anion] = counts.get(anion, 0) + salt_mol_count

    # 修正：若总分子数有微小偏差（四舍五入导致），补到占比最大的组分
    actual_total = sum(counts.values())
    if actual_total != total_molecules:
        diff = total_molecules - actual_total
        # 找到占比最大的组分，补/减差值
        max_comp = max(counts, key=counts.get)
        counts[max_comp] += diff

    return counts


import json
import os
from typing import Dict, Optional, Tuple, List
from datetime import datetime

def split_salt_smiles(salt_smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    拆分盐的复合SMILES为阳离子和阴离子的独立SMILES
    核心处理：[Li+].F[P-](F)(F)(F)(F)F → [Li+] 和 F[P-](F)(F)(F)(F)F
    """
    if not salt_smiles or not isinstance(salt_smiles, str):
        print("❌ SMILES字符串为空或格式错误")
        return None, None
    
    # 按.拆分离子片段，去除空字符串和空格
    ion_fragments = [frag.strip() for frag in salt_smiles.split('.') if frag.strip()]
    if len(ion_fragments) == 0:
        print("❌ 拆分后无有效离子片段")
        return None, None
    
    cation_smiles = None
    anion_smiles = None
    
    # 识别阳离子(含+)和阴离子(含-)
    for frag in ion_fragments:
        if '+' in frag and '-' not in frag:
            cation_smiles = frag
        elif '-' in frag and '+' not in frag:
            anion_smiles = frag
        elif '+' in frag and '-' in frag:
            print(f"⚠️ 片段{frag}同时含+/-电荷，无法识别阴阳离子")
    
    print(f"✅ SMILES拆分完成：阳离子={cation_smiles} | 阴离子={anion_smiles}")
    return cation_smiles, anion_smiles

def validate_electrolyte_standard_dict(elec_dict: Dict) -> bool:
    """
    严格验证你定义的标准化字典结构，确保字段不缺失、数值合法
    """
    # 1. 验证顶层核心模块
    required_top_level = ["metadata", "components", "box_params", "simulation_params"]
    for module in required_top_level:
        if module not in elec_dict:
            raise KeyError(f"缺失顶层核心模块：{module}")
    
    # 2. 验证metadata字段
    required_metadata = ["recipe_name", "version", "create_time"]
    for field in required_metadata:
        if field not in elec_dict["metadata"]:
            raise KeyError(f"metadata模块缺失字段：{field}")
    
    # 3. 验证components字段（核心：质量分数+SMILES+摩尔质量）
    required_components_types = ["solvent", "salt", "additive"]
    for comp_type in required_components_types:
        if comp_type not in elec_dict["components"]:
            raise KeyError(f"components模块缺失组分类型：{comp_type}")
        
        # 验证每个组分的必选字段
        for comp in elec_dict["components"][comp_type]:
            required_comp_fields = ["name", "full_name", "smiles", "mass_fraction", "molar_mass", "purity"]
            for field in required_comp_fields:
                if field not in comp:
                    raise KeyError(f"{comp_type}组分 {comp.get('name', '未知')} 缺失字段：{field}")
            
            # 数值合法性校验
            if comp["mass_fraction"] < 0 or comp["mass_fraction"] > 1:
                raise ValueError(f"{comp_type}组分 {comp['name']} 质量分数需在0~1之间")
            if comp["molar_mass"] <= 0:
                raise ValueError(f"{comp_type}组分 {comp['name']} 摩尔质量必须大于0")
            if comp["purity"] < 0 or comp["purity"] > 1:
                raise ValueError(f"{comp_type}组分 {comp['name']} 纯度需在0~1之间")
    
    # 4. 验证质量分数总和≈1（允许±0.1%浮点误差）
    total_mass_fraction = 0.0
    for comp_type in required_components_types:
        for comp in elec_dict["components"][comp_type]:
            total_mass_fraction += comp["mass_fraction"]
    if not (0.999 <= total_mass_fraction <= 1.001):
        raise ValueError(f"所有组分质量分数总和为 {round(total_mass_fraction, 6)}，需近似等于1（当前总和偏离过大）")
    
    # 5. 验证box_params字段
    required_box_fields = ["target_length", "length_unit", "box_type"]
    for field in required_box_fields:
        if field not in elec_dict["box_params"]:
            raise KeyError(f"box_params模块缺失字段：{field}")
    if elec_dict["box_params"]["target_length"] <= 0:
        raise ValueError("盒子目标边长必须大于0")
    if elec_dict["box_params"]["length_unit"] not in ["nm", "Å", "cm"]:
        raise ValueError("盒子边长单位仅支持nm/Å/cm")
    
    # 6. 验证simulation_params字段
    required_sim_fields = ["temperature", "pressure", "unit"]
    for field in required_sim_fields:
        if field not in elec_dict["simulation_params"]:
            raise KeyError(f"simulation_params模块缺失字段：{field}")
    required_unit_fields = ["temperature", "pressure", "concentration"]
    for field in required_unit_fields:
        if field not in elec_dict["simulation_params"]["unit"]:
            raise KeyError(f"simulation_params.unit模块缺失字段：{field}")
    
    print("✅ 标准化字典结构验证通过")
    return True


def auto_generate_solute_to_ions(salt_names: List[str]) -> Dict[str, Tuple[str, str]]:
    """
    自动生成盐名称到离子简写的映射（替代手动定义）
    输入：盐名称列表（如['LiPF6', 'LiFSI']）
    输出：自动生成的映射字典（如{'LiPF6': ('LI', 'PF6'), ...}）
    """
    # 基础映射规则（覆盖常见阳离子/阴离子）
    cation_map = {
        'Li': 'LI', 'Na': 'NA', 'K': 'K', 'Mg': 'MG', 'Ca': 'CA',  # 阳离子元素→简写
        'Lithium': 'LI', 'Sodium': 'NA', 'Potassium': 'K'  # 全称兼容
    }
    anion_map = {
        'PF6': 'PF6', 'FSI': 'FSI', 'TFSI': 'TFSI', 'DFOB': 'DFOB', 
        'ClO4': 'CLO4', 'BF4': 'BF4', 'SO4': 'SO4', 'Cl': 'CL', 'Br': 'BR'
    }
    
    solute_to_ions = {}
    # 正则匹配：提取阳离子（元素符号/全称）和阴离子
    cation_pattern = re.compile(r'^(' + '|'.join(cation_map.keys()) + r')', re.IGNORECASE)
    
    for salt_name in salt_names:
        salt_name_clean = salt_name.strip()
        if not salt_name_clean:
            continue
        
        # 步骤1：匹配阳离子
        cation_match = cation_pattern.match(salt_name_clean)
        if not cation_match:
            print(f"⚠️ 无法识别盐{salt_name}的阳离子，跳过")
            continue
        cation_raw = cation_match.group(1).capitalize()  # 统一首字母大写（Li/Na）
        cation_abbr = cation_map[cation_raw]
        
        # 步骤2：提取阴离子（去掉阳离子部分后的剩余内容）
        anion_raw = salt_name_clean[cation_match.end():]
        # 匹配阴离子简写（不区分大小写）
        anion_abbr = None
        for anion_key in anion_map.keys():
            if anion_raw.upper() == anion_key:
                anion_abbr = anion_map[anion_key]
                break
        
        if not anion_abbr:
            print(f"⚠️ 无法识别盐{salt_name}的阴离子{anion_raw}，跳过")
            continue
        
        # 步骤3：生成映射
        solute_to_ions[salt_name_clean] = (cation_abbr, anion_abbr)
        print(f"✅ 自动生成映射：{salt_name_clean} → ({cation_abbr}, {anion_abbr})")
    
    return solute_to_ions

def calculate_md_params_from_standard_dict(elec_dict: Dict, exp_density: float = 1.20) -> Dict:
    """
    补全MD模拟参数计算逻辑，集成盐SMILES拆分功能
    """
    # 1. 基础参数提取
    target_length = elec_dict["box_params"]["target_length"]  # nm
    avogadro = 6.02214076e23  # 阿伏伽德罗常数
    
    # 2. 计算目标盒子体积 & 体系总质量
    target_volume_nm3 = target_length** 3  # 目标体积(nm³)
    target_volume_cm3 = target_volume_nm3 / 1e21  # 转换为cm³（1cm³=1e21nm³）
    total_system_mass_g = exp_density * target_volume_cm3  # 体系总质量(g)
    
    # 3. 遍历组分计算：质量→摩尔数→分子数，盐组分拆分SMILES
    calculated_components = []
    total_moles = 0.0
    
    for comp_type in ["solvent", "salt", "additive"]:
        comp_list = elec_dict["components"][comp_type]
        if len(comp_list) == 0:
            continue
        
        for comp in comp_list:
            # 计算组分质量、摩尔数、分子数
            comp_mass_g = total_system_mass_g * comp["mass_fraction"]
            comp_moles = comp_mass_g / comp["molar_mass"]
            comp_mol_number = int(round(comp_moles * avogadro))
            
            # 复制原组分信息，新增计算字段
            comp_calc = comp.copy()
            comp_calc.update({
                "component_mass_g": round(comp_mass_g, 8),
                "component_moles": round(comp_moles, 8),
                "molecule_number": comp_mol_number,
                "component_type": comp_type
            })
            
            # 盐组分拆分SMILES
            if comp_type == "salt":
                cation_smiles, anion_smiles = split_salt_smiles(comp["smiles"])
                comp_calc["cation_smiles"] = cation_smiles
                comp_calc["anion_smiles"] = anion_smiles
            
            calculated_components.append(comp_calc)
            total_moles += comp_moles
    
    # 4. 计算摩尔分数
    for comp in calculated_components:
        comp["mole_fraction"] = round(comp["component_moles"] / total_moles, 6) if total_moles > 0 else 0.0
    
    # 5. 计算盒子实际尺寸（修正密度偏差）
    actual_volume_cm3 = total_system_mass_g / exp_density
    actual_volume_nm3 = actual_volume_cm3 * 1e21
    actual_length_nm = round(actual_volume_nm3 **(1/3), 4)
    
    # 6. 组装结果（保留原结构+新增计算字段）
    result_dict = elec_dict.copy()
    result_dict["calculated_params"] = {
        "exp_density_used": exp_density,
        "total_system_mass_g": round(total_system_mass_g, 8),
        "total_system_moles": round(total_moles, 8),
        "target_box_volume_nm3": round(target_volume_nm3, 4),
        "actual_box_volume_nm3": round(actual_volume_nm3, 4),
        "actual_box_length_nm": actual_length_nm,
        "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "components_calculated": calculated_components
    }
    
    print(f"✅ 配方 {elec_dict['metadata']['recipe_name']} MD参数计算完成")
    return result_dict

def load_standard_electrolyte_recipe(json_path: str, exp_density: float = 1.20) -> Optional[List[Dict]]:
    """
    标准化流程：读取JSON → 验证结构 → 计算MD参数（适配多配方）
    返回：多配方计算结果列表，失败返回None
    """
    # 1. 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"❌ 错误：文件 {json_path} 不存在")
        return None
    
    # 2. 读取JSON文件
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            elec_dicts = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON格式解析错误：{e}")
        return None
    except Exception as e:
        print(f"❌ 文件读取未知错误：{e}")
        return None
    
    # 验证是否包含recipes字段
    if "recipes" not in elec_dicts or not isinstance(elec_dicts["recipes"], list):
        print("❌ JSON文件缺少recipes列表，不符合多配方格式")
        return None
    
    result_dicts = []
    # 遍历每个配方处理
    for elec_dict in elec_dicts['recipes']: 
        # 3. 验证字典结构
        try:
            validate_electrolyte_standard_dict(elec_dict)
        except (KeyError, ValueError) as e:
            print(f"❌ 配方 {elec_dict.get('metadata', {}).get('recipe_name', '未知')} 验证失败：{e}")
            return None
        
        # 4. 计算MD模拟参数（集成SMILES拆分）
        try:
            result_dict = calculate_md_params_from_standard_dict(elec_dict, exp_density)
            result_dicts.append(result_dict)
        except Exception as e:
            print(f"❌ 配方 {elec_dict.get('metadata', {}).get('recipe_name', '未知')} MD参数计算失败：{e}")
            return None
    
    print(f"\n✅ 所有配方处理完成，共成功处理 {len(result_dicts)} 个配方")
    return result_dicts 


if __name__ == "__main__":
    json_path = 'standard_electrolyte_recipe.json'
    # read electrolyte formula data 
    recipe_results = load_standard_electrolyte_recipe(json_path)

    # check smiles and molecular inventory
    mapping_smiles_list = {}
    for recipe_result in recipe_results:
        components = recipe_result['calculated_params']['components_calculated']
        for component in components:
            if component['component_type'] == 'salt':
                solute_to_ions = auto_generate_solute_to_ions([component['name']])
                cation_name, anion_name = solute_to_ions[component['name']]
                component['cation_name'] = cation_name
                component['anion_name'] = anion_name

                mapping_smiles_list[cation_name] = component['cation_smiles']
                mapping_smiles_list[anion_name] = component['anion_smiles']
            else:
                mapping_smiles_list[component['name']] = component['smiles']
                # mapping_smiles_list[mol['smiles']] = mol['name']

    print(mapping_smiles_list)

    MOLECULE_DB = {}
    for name, smiles in mapping_smiles_list.items():
        try:
            molar_mass = smiles_to_molar_mass(smiles)
        except Exception as e:
            print(f"⚠️ Error for {name}: {e}")
            continue
        MOLECULE_DB[name] = (round(molar_mass, 2))

    # 去重
    # mapping_smiles_list = set(mapping_smiles_list)
    print(mapping_smiles_list)
        
    # 检查smiles是否存在，或者不检查？
    from rdkit import Chem
    from rdkit.Chem import inchi

    def smiles_to_inchikey(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return inchi.MolToInchiKey(mol) if mol else None

    key_unparamed_list = {}
    key_mapped_list = {}
    for name in mapping_smiles_list:
        smiles = mapping_smiles_list[name]
        query_key = smiles_to_inchikey(smiles)
        
        found_keys = []
        for k, v in SMILES_DICT.items():
            if smiles_to_inchikey(v) == query_key:
                found_keys.append(k)
        
        if found_keys:
            print("exist, keys:", found_keys)
            key_mapped_list[name] = found_keys[0]
        else:
            key_unparamed_list[name] = smiles
            print("not exist:", smiles)

    for name in key_unparamed_list:
        smiles = key_unparamed_list[name]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i+1)
        mapped_smiles = Chem.MolToSmiles(mol, canonical=True)
        key_unparamed_list[name] = mapped_smiles
    
    print(key_unparamed_list)
    # SMILES_DICT.update(mapping_smiles_list)

    # create molecular params 
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

    OUTPUT_DIR = os.path.abspath("./params_results")

    def write_gro(mol: Molecule, save_path: str):
        atoms_gro = mol.conformers[0].to_ase_atoms()
        atoms_gro.set_array('residuenames', np.array([mol.name] * mol.natoms))
        aio.write(save_path, atoms_gro)


    def gro2pdb_fix_box(gro_file: str, output_dir: str, mol, box_size: float = 60.0):
        """
        简洁版：修复GRO盒子尺寸并转换为PDB，设置残基名
        """
        # 修复GRO盒子（Å转nm）
        box_nm = box_size / 10.0
        with open(gro_file, 'r') as f:
            lines = f.readlines()
        last_line = lines[-1].strip().split()
        if len(last_line) not in [3, 9]:
            lines[-1] = f"{box_nm:.5f} {box_nm:.5f} {box_nm:.5f}\n"
            with open(gro_file, 'w') as f:
                f.writelines(lines)
        
        # GRO转PDB并设置残基名
        pdb_file = f"{output_dir}/{mol.name}.pdb"
        u = mda.Universe(gro_file)
        for res in u.residues:
            res.resname = mol.name
        with mda.Writer(pdb_file, multiframe=False) as w:
            w.write(u.atoms)

    # load model
    model_dir = get_data_file_path('optimal.pt', 'byteff2.trained_models')
    model = load_model(os.path.dirname(model_dir))
    # generate input mol
    mols = all_name_mapped_smiles

    import os
    import MDAnalysis as mda
    from MDAnalysis.coordinates import PDB
    import numpy as np
    import glob

    for name in key_unparamed_list.keys():
        # for name in five_mapped_smiles.keys():
        mps = key_unparamed_list[name]
        print(name, mps)

        # mps = mols['FSI']
        OUTPUT_DIR = os.path.abspath(f"./template/input_data/{name}")
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
                    
                    # need to fix PDB data to pack:
                    gro2pdb_fix_box(f'{OUTPUT_DIR}/{mol.name}.gro', OUTPUT_DIR, mol)

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


    # create workdir 
    WORK_DIR = '/AI4S/Projects/CAFF/ELF_MD_SOP/test_workdir'
    template_dir = '/AI4S/Projects/CAFF/ELF_MD_SOP/template'
    os.makedirs(WORK_DIR, exist_ok=True)

    for recipe_result in recipe_results:
        print('create formula workdir ')
        mass_fraction_dict = recipe_result['calculated_params']['components_calculated'] 
        counts = mass_fraction_to_counts(mass_fraction_dict, total_molecules=1000)
        recipe_result['calculated_params']['mol_counts'] = counts
        recipe_name =  recipe_result['metadata']['recipe_name']
        folder = f"{WORK_DIR}/recipe_{recipe_name}"
        mol_counts = recipe_result['calculated_params']['mol_counts']
        temperature = recipe_result['simulation_params']['temperature']
        if os.path.exists(folder):
            continue
        else:
            os.makedirs(folder, exist_ok=True)
            mol_masses = get_mol_masses()
            save_run_file(folder, mol_masses, mol_counts, temperature=temperature, template_dir=template_dir)
            save_pack_file(folder, mol_counts, template_dir=template_dir)
            save_topo_file(folder, mol_counts, template_dir=template_dir)
            # need a excuate packmol process!         
