# Electrolyte Bulk MD Simulation SOP
"""
优化说明（2025-12-31）：
- 在之前优化版本基础上，增加对输入 JSON 的通配符支持（glob pattern）。
- 添加 argparse：允许通过命令行传入 JSON 模式、density、workdir、template_dir、以及 --verbose。
- 行为：脚本会把匹配到的所有 JSON 文件依次加载并合并配方结果，保留原有的单文件兼容性。
- 若没有匹配到任何文件，脚本会给出友好提示并退出。
- 其它逻辑与函数保持不变（仅在主流程中汇聚多个文件的结果并读取 CLI 参数）。
"""

import argparse
import os
import re
import glob
import shutil
import logging
import traceback
import json
from datetime import datetime
from functools import lru_cache
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, inchi

# domain inventory (保持原始导入)
from mol_inventory import *  # noqa: F401,F403

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

# ----------------- Logging 配置（默认 INFO） -----------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ========== PDB 质量提取 ==========
def get_mol_masses(mol_counts, template_dir):
    def parse_pdb_for_masses(pdb_file):
        masses = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    element = line[76:78].strip().upper()
                    if not element:
                        atom_name = line[12:16].strip()
                        elem = ''.join([c for c in atom_name if c.isalpha()])
                        element = elem[:2].upper() if len(elem) >= 2 else elem.upper()
                    if element in element_to_mass:
                        masses.append(element_to_mass[element])
        return masses

    mol_masses = {}
    for key in mol_counts:
        pdb_file = f'{template_dir}/input_data/{key}/{key}.pdb'
        mol_name = os.path.splitext(os.path.basename(pdb_file))[0]
        if not os.path.exists(pdb_file):
            logging.warning("PDB 文件不存在，跳过：%s", pdb_file)
            continue
        masses = parse_pdb_for_masses(pdb_file)
        if masses:
            mol_masses[mol_name] = masses
    return mol_masses


# ===================== SMILES 映射优化（缓存 + 预构建） =====================
def smiles_to_inchikey(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return inchi.MolToInchiKey(mol) if mol else None


@lru_cache(maxsize=None)
def cached_smiles_to_inchikey(smiles: str) -> Optional[str]:
    try:
        return smiles_to_inchikey(smiles)
    except Exception as e:
        logging.warning("SMILES 转 InChIKey 失败 (%s): %s", smiles, e)
        return None


def generate_mapped_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning("无法解析 SMILES：%s", smiles)
            return None
        mol = Chem.AddHs(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        logging.warning("生成映射 SMILES 失败 (%s): %s", smiles, e)
        return None


def optimize_smiles_mapping(mapping_smiles_list: dict, SMILES_DICT: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    优化后的SMILES映射逻辑
    返回：(mapped_smiles_dict, found_keys_dict)
    """
    # 1. 预构建 InChIKey -> (k, v) 映射表
    inchikey_to_kv: Dict[str, Tuple[str, str]] = {}
    for k, v in SMILES_DICT.items():
        inchikey = cached_smiles_to_inchikey(v)
        if inchikey:
            inchikey_to_kv[inchikey] = (k, v)

    mapped_smiles_dict: Dict[str, str] = {}
    found_keys_dict: Dict[str, str] = {}

    for name, smiles in mapping_smiles_list.items():
        query_inchikey = cached_smiles_to_inchikey(smiles)
        if not query_inchikey:
            continue

        if query_inchikey in inchikey_to_kv:
            k, v = inchikey_to_kv[query_inchikey]
            logging.info("匹配到已有键值：%s -> %s （%s）", name, k, v)
            found_keys_dict[name] = k
            mapped_smiles_dict[name] = v
        else:
            mapped_smiles = generate_mapped_smiles(smiles)
            if mapped_smiles:
                mapped_smiles_dict[name] = mapped_smiles

    return mapped_smiles_dict, found_keys_dict


# ========== 文件生成 ==========
def save_run_file(folder, mol_masses, mol_counts, temperature=298,
                  template_dir='/AI4S/Projects/CAFF/ELF_MD_SOP/template'):
    template_py = f"{template_dir}/run_transport.py"
    output_py = os.path.join(folder, "run_md.py")
    if not os.path.exists(template_py):
        logging.error("模板脚本缺失：%s", template_py)
        raise FileNotFoundError(template_py)
    with open(template_py, 'r') as f:
        content = f.read()
    content = re.sub(r'TEMPERATURE\s*=\s*\d+', f'TEMPERATURE = {temperature}', content)

    number_dict_str = ", ".join([f'"{k}": {v}' for k, v in mol_counts.items()])
    content = re.sub(r'species_number_dict = \{.*?\}', f'species_number_dict = {{{number_dict_str}}}', content,
                     flags=re.DOTALL)
    charge_dict_str = ", ".join([f'"{k}": {mol_charges.get(k, 0)}' for k in mol_counts])
    content = re.sub(r'species_charges_dict = \{.*?\}', f'species_charges_dict = {{{charge_dict_str}}}', content,
                     flags=re.DOTALL)
    mass_dict_lines = [f'    "{mol}": [{", ".join(str(m) for m in mol_masses.get(mol, [12.0]))}]' for mol in mol_counts]
    mass_dict_full = "species_mass_dict = {\n" + ",\n".join(mass_dict_lines) + "\n}"
    content = re.sub(r'species_mass_dict = \{.*?\}', mass_dict_full, content, flags=re.DOTALL)
    mol_list_str = ", ".join([f"'{k}'" for k in mol_counts])
    content = re.sub(r"for mol in \[.*?\]:", f"for mol in [{mol_list_str}]:", content)
    with open(output_py, 'w') as f:
        f.write(content)


def save_pack_file(folder, mol_counts, template_dir='/AI4S/Projects/CAFF/ELF_MD_SOP/template'):
    template_pack = f"{template_dir}/input_data/pack_data.py"
    output_pack = os.path.join(folder, "prepare_system.py")
    if not os.path.exists(template_pack):
        logging.error("pack_data 模板缺失：%s", template_pack)
        raise FileNotFoundError(template_pack)
    with open(template_pack, 'r') as f:
        pack_content = f.read()
    mol_entries = [f"    '{template_dir}/input_data/{mol}/{mol}.gro': {{'resname': '{mol}', 'count': {count}}},"
                   for mol, count in mol_counts.items()]
    mol_dict_str = "{\n" + "\n".join(mol_entries) + "\n}"
    pack_content = re.sub(r"molecules = \{.*?\}", f"molecules = {mol_dict_str}", pack_content, flags=re.DOTALL)
    # 固定 box_size（保留原注释逻辑）
    box_size = 150.0
    pack_content = re.sub(r"box_size = .*?#", f"box_size = {box_size:.1f}  #", pack_content)
    with open(output_pack, 'w') as f:
        f.write(pack_content)


def save_topo_file(folder, mol_counts, template_dir='/AI4S/Projects/CAFF/ELF_MD_SOP/template'):
    topol_content = "[ defaults ]\n1 3 yes 0.5 0.5\n\n[ atomtypes ]\n"
    for mol in mol_counts:
        topol_content += f'#include "{template_dir}/input_data/{mol}/{mol}.atp"\n'
    for mol in mol_counts:
        topol_content += f'#include "{template_dir}/input_data/{mol}/{mol}.itp"\n'
    topol_content += "\n[ system ]\n" + " ".join(mol_counts.keys()) + "\n\n[molecules]\n"
    for mol, count in mol_counts.items():
        topol_content += f"{mol} {count}\n"
    with open(os.path.join(folder, "topol.top"), 'w') as f:
        f.write(topol_content)


# ========== 分子数分配 ==========
def mass_fraction_to_counts(mass_fraction_list: List[Dict], total_molecules=10000,
                            solute_to_ions: Optional[Dict[str, Tuple[str, str]]] = None) -> Dict[str, int]:
    """
    基于总分子数计算各组分数量（mass_fraction_list 是 components_calculated 列表）
    增强：可选传入 solute_to_ions 映射；若未传入，则尝试从 salt 组分字段读取 cation_name/anion_name。
    返回：{name: count}
    """
    NA_local = 6.02214076e23  # 保留注释一致性

    # 1. 分离溶剂/盐类
    solvents = [component for component in mass_fraction_list if component['component_type'] != 'salt']
    salts = [component for component in mass_fraction_list if component['component_type'] == 'salt']

    # 2. 归一化质量分数（确保总和为1）
    total_mass = sum([component['mass_fraction'] for component in mass_fraction_list])
    if total_mass <= 0:
        raise ValueError("质量分数总和为 0 或负值")
    weight_fractions = {component['name']: component['mass_fraction'] / total_mass for component in mass_fraction_list}

    # 3. 质量分数 -> 摩尔当量
    mole_equivalent = {}
    total_mole_equivalent = 0.0

    for solvent in solvents:
        name = solvent['name']
        mol_mass = solvent['molar_mass']
        mole_equivalent[name] = weight_fractions[name] / mol_mass
        total_mole_equivalent += mole_equivalent[name]

    for salt in salts:
        name = salt['name']
        salt_mol_mass = salt['molar_mass']
        mole_equivalent[name] = weight_fractions[name] / salt_mol_mass
        total_mole_equivalent += mole_equivalent[name]

    if total_mole_equivalent <= 0:
        raise ValueError("总摩尔当量 <= 0，无法分配分子数")

    mole_fractions = {
        component['name']: mole_equivalent[component['name']] / total_mole_equivalent
        for component in solvents + salts
    }

    counts: Dict[str, int] = {}

    # 溶剂分子数
    for solvent in solvents:
        name = solvent['name']
        counts[name] = int(round(total_molecules * mole_fractions[name]))

    # 盐拆分为阴阳离子
    for salt in salts:
        name = salt['name']
        salt_mol_count = int(round(total_molecules * mole_fractions[name]))

        # 优先使用传入的 solute_to_ions 映射
        if solute_to_ions and name in solute_to_ions:
            cation_abbr, anion_abbr = solute_to_ions[name]
        else:
            # 尝试从 salt 本身字段读取（调用方在 calculate_md_params_from_standard_dict 中可能已写入）
            cation_abbr = salt.get('cation_name')
            anion_abbr = salt.get('anion_name')

        if not cation_abbr or not anion_abbr:
            raise KeyError(f"盐 {name} 的离子映射未找到（请传入 solute_to_ions 或在配方中提供 cation_name/anion_name）")

        counts[cation_abbr] = counts.get(cation_abbr, 0) + salt_mol_count
        counts[anion_abbr] = counts.get(anion_abbr, 0) + salt_mol_count

    # 修正四舍五入误差
    actual_total = sum(counts.values())
    if actual_total != total_molecules:
        diff = total_molecules - actual_total
        max_comp = max(counts, key=counts.get)
        counts[max_comp] += diff

    return counts


# ========== SMILES & 配方处理 ==========
def split_salt_smiles(salt_smiles: str) -> Tuple[Optional[str], Optional[str]]:
    if not salt_smiles or not isinstance(salt_smiles, str):
        logging.warning("SMILES 字符串为空或格式错误")
        return None, None

    ion_fragments = [frag.strip() for frag in salt_smiles.split('.') if frag.strip()]
    if len(ion_fragments) == 0:
        logging.warning("拆分后无有效离子片段")
        return None, None

    cation_smiles = None
    anion_smiles = None

    for frag in ion_fragments:
        if '+' in frag and '-' not in frag:
            cation_smiles = frag
        elif '-' in frag and '+' not in frag:
            anion_smiles = frag
        elif '+' in frag and '-' in frag:
            logging.warning("片段 %s 同时含 +/-，无法识别", frag)

    logging.info("SMILES 拆分：阳离子=%s | 阴离子=%s", cation_smiles, anion_smiles)
    return cation_smiles, anion_smiles


def validate_electrolyte_standard_dict(elec_dict: Dict) -> bool:
    required_top_level = ["metadata", "components", "box_params", "simulation_params"]
    for module in required_top_level:
        if module not in elec_dict:
            raise KeyError(f"缺失顶层核心模块：{module}")

    required_metadata = ["recipe_name", "version", "create_time"]
    for field in required_metadata:
        if field not in elec_dict["metadata"]:
            raise KeyError(f"metadata模块缺失字段：{field}")

    required_components_types = ["solvent", "salt", "additive"]
    for comp_type in required_components_types:
        if comp_type not in elec_dict["components"]:
            raise KeyError(f"components模块缺失组分类型：{comp_type}")
        for comp in elec_dict["components"][comp_type]:
            required_comp_fields = ["name", "full_name", "smiles", "mass_fraction", "molar_mass", "purity"]
            for field in required_comp_fields:
                if field not in comp:
                    raise KeyError(f"{comp_type}组分 {comp.get('name', '未知')} 缺失字段：{field}")
            if comp["mass_fraction"] < 0 or comp["mass_fraction"] > 1:
                raise ValueError(f"{comp_type}组分 {comp['name']} 质量分数需在0~1之间")
            if comp["molar_mass"] <= 0:
                raise ValueError(f"{comp_type}组分 {comp['name']} 摩尔质量必须大于0")
            if comp["purity"] < 0 or comp["purity"] > 1:
                raise ValueError(f"{comp_type}组分 {comp['name']} 纯度需在0~1之间")

    total_mass_fraction = 0.0
    for comp_type in required_components_types:
        for comp in elec_dict["components"][comp_type]:
            total_mass_fraction += comp["mass_fraction"]
    if not (0.999 <= total_mass_fraction <= 1.001):
        raise ValueError(f"所有组分质量分数总和为 {round(total_mass_fraction, 6)}，需近似等于1")

    required_box_fields = ["target_length", "length_unit", "box_type"]
    for field in required_box_fields:
        if field not in elec_dict["box_params"]:
            raise KeyError(f"box_params模块缺失字段：{field}")
    if elec_dict["box_params"]["target_length"] <= 0:
        raise ValueError("盒子目标边长必须大于0")
    if elec_dict["box_params"]["length_unit"] not in ["nm", "Å", "cm"]:
        raise ValueError("盒子边长单位仅支持nm/Å/cm")

    required_sim_fields = ["temperature", "pressure", "unit"]
    for field in required_sim_fields:
        if field not in elec_dict["simulation_params"]:
            raise KeyError(f"simulation_params模块缺失字段：{field}")
    required_unit_fields = ["temperature", "pressure", "concentration"]
    for field in required_unit_fields:
        if field not in elec_dict["simulation_params"]["unit"]:
            raise KeyError(f"simulation_params.unit模块缺失字段：{field}")

    logging.info("标准化字典结构验证通过：%s", elec_dict["metadata"].get("recipe_name"))
    return True


def auto_generate_solute_to_ions(salt_names: List[str]) -> Dict[str, Tuple[str, str]]:
    cation_map = {
        'Li': 'LI', 'Na': 'NA', 'K': 'K', 'Mg': 'MG', 'Ca': 'CA',
        'Lithium': 'LI', 'Sodium': 'NA', 'Potassium': 'K'
    }
    anion_map = {
        'PF6': 'PF6', 'FSI': 'FSI', 'TFSI': 'TFSI', 'DFOB': 'DFOB',
        'ClO4': 'CLO4', 'BF4': 'BF4', 'SO4': 'SO4', 'Cl': 'CL', 'Br': 'BR'
    }

    solute_to_ions: Dict[str, Tuple[str, str]] = {}
    cation_pattern = re.compile(r'^(' + '|'.join(cation_map.keys()) + r')', re.IGNORECASE)

    for salt_name in salt_names:
        salt_name_clean = salt_name.strip()
        if not salt_name_clean:
            continue
        cation_match = cation_pattern.match(salt_name_clean)
        if not cation_match:
            logging.warning("无法识别盐 %s 的阳离子，跳过", salt_name)
            continue
        cation_raw = cation_match.group(1).capitalize()
        cation_abbr = cation_map.get(cation_raw)
        if not cation_abbr:
            logging.warning("无法映射阳离子：%s", cation_raw)
            continue
        anion_raw = salt_name_clean[cation_match.end():]
        anion_abbr = None
        for anion_key in anion_map.keys():
            if anion_raw.upper() == anion_key:
                anion_abbr = anion_map[anion_key]
                break
        if not anion_abbr:
            logging.warning("无法识别盐 %s 的阴离子 %s，跳过", salt_name, anion_raw)
            continue
        solute_to_ions[salt_name_clean] = (cation_abbr, anion_abbr)
        logging.info("自动生成映射：%s -> (%s, %s)", salt_name_clean, cation_abbr, anion_abbr)

    return solute_to_ions


def calculate_md_params_from_standard_dict(elec_dict: Dict, exp_density: float = 1.20) -> Dict:
    target_length = elec_dict["box_params"]["target_length"]
    avogadro = 6.02214076e23

    target_volume_nm3 = target_length ** 3
    target_volume_cm3 = target_volume_nm3 / 1e21
    total_system_mass_g = exp_density * target_volume_cm3

    calculated_components = []
    total_moles = 0.0

    for comp_type in ["solvent", "salt", "additive"]:
        comp_list = elec_dict["components"][comp_type]
        if len(comp_list) == 0:
            continue

        for comp in comp_list:
            comp_mass_g = total_system_mass_g * comp["mass_fraction"]
            comp_moles = comp_mass_g / comp["molar_mass"]
            comp_mol_number = int(round(comp_moles * avogadro))

            comp_calc = comp.copy()
            comp_calc.update({
                "component_mass_g": round(comp_mass_g, 8),
                "component_moles": round(comp_moles, 8),
                "molecule_number": comp_mol_number,
                "component_type": comp_type
            })

            if comp_type == "salt":
                cation_smiles, anion_smiles = split_salt_smiles(comp["smiles"])
                comp_calc["cation_smiles"] = cation_smiles
                comp_calc["anion_smiles"] = anion_smiles

            calculated_components.append(comp_calc)
            total_moles += comp_moles

    for comp in calculated_components:
        comp["mole_fraction"] = round(comp["component_moles"] / total_moles, 6) if total_moles > 0 else 0.0

    actual_volume_cm3 = total_system_mass_g / exp_density
    actual_volume_nm3 = actual_volume_cm3 * 1e21
    actual_length_nm = round(actual_volume_nm3 ** (1 / 3), 4)

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

    logging.info("配方 %s MD参数计算完成", elec_dict['metadata'].get('recipe_name'))
    return result_dict


def load_standard_electrolyte_recipe(json_path: str, exp_density: float = 1.20) -> Optional[List[Dict]]:
    """
    读取 JSON 并对每个配方单独处理；失败的配方会被记录并跳过，最终返回成功处理的配方列表。
    """
    if not os.path.exists(json_path):
        logging.error("文件不存在：%s", json_path)
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            elec_dicts = json.load(f)
    except json.JSONDecodeError as e:
        logging.error("JSON 解析错误：%s", e)
        return None
    except Exception as e:
        logging.error("读取文件失败：%s", e)
        return None

    if "recipes" not in elec_dicts or not isinstance(elec_dicts["recipes"], list):
        logging.error("JSON 文件缺少 recipes 列表")
        return None

    result_dicts: List[Dict] = []
    failures: List[Tuple[str, str]] = []

    for elec_dict in elec_dicts['recipes']:
        recipe_name = elec_dict.get('metadata', {}).get('recipe_name', '未知')
        try:
            validate_electrolyte_standard_dict(elec_dict)
        except (KeyError, ValueError) as e:
            logging.error("配方 %s 验证失败：%s", recipe_name, e)
            failures.append((recipe_name, f"validate_error: {e}"))
            continue

        try:
            result_dict = calculate_md_params_from_standard_dict(elec_dict, exp_density)
            result_dicts.append(result_dict)
        except Exception as e:
            logging.error("配方 %s MD 参数计算失败：%s", recipe_name, e)
            traceback.print_exc()
            failures.append((recipe_name, f"calc_error: {e}"))
            continue

    logging.info("配方处理完成：成功 %d，失败 %d", len(result_dicts), len(failures))
    if failures:
        logging.warning("失败配方列表：%s", failures)
    return result_dicts


# ========== 模板检查 ==========
def check_template_files(template_dir: str) -> bool:
    """检查常用模板文件/目录是否存在，返回 True 表示一切OK"""
    required = [
        os.path.join(template_dir, 'run_transport.py'),
        os.path.join(template_dir, 'input_data', 'pack_data.py'),
        os.path.join(template_dir, 'input_data')
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        logging.error("模板目录不完整，缺少：%s", missing)
        return False
    logging.info("模板检查通过：%s", template_dir)
    return True


# ---------------------------
# 主体运行流程（增加 argparse 支持 JSON 通配符）
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MD workdirs from standard electrolyte recipe JSON(s)")
    parser.add_argument("--json", "-j", dest="json_pattern", default="standard_electrolyte_recipe.json",
                        help="输入 JSON 路径或通配符（例如 'standard_electrolyte_recipe.json' 或 'recipes/*.json'）")
    parser.add_argument("--density", "-d", dest="density", type=float, default=1.20,
                        help="体系密度 g/cm^3（默认 1.20）")
    parser.add_argument("--workdir", "-w", dest="workdir", default="/AI4S/Projects/CAFF/ELF_MD_SOP/test_workdir",
                        help="输出工作目录（默认 /AI4S/Projects/CAFF/ELF_MD_SOP/test_workdir）")
    parser.add_argument("--template", "-t", dest="template_dir", default="/AI4S/Projects/CAFF/ELF_MD_SOP/template",
                        help="模板目录（默认 /AI4S/Projects/CAFF/ELF_MD_SOP/template）")
    parser.add_argument("--verbose", "-v", dest="verbose", action="store_true",
                        help="启用详细日志（DEBUG）")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("启用详细日志模式（DEBUG）")

    json_pattern = args.json_pattern
    exp_density = args.density
    WORK_DIR = os.path.abspath(args.workdir)
    template_dir = args.template_dir

    # 使用 glob 查找匹配的文件
    matched_files = sorted(glob.glob(json_pattern))
    if not matched_files:
        logging.error("未找到匹配的 JSON 文件（pattern=%s）。请确认路径或使用通配符。", json_pattern)
        raise SystemExit(1)

    # 合并来自所有匹配文件的配方结果
    recipe_results: List[Dict] = []
    for jf in matched_files:
        logging.info("加载配方文件：%s", jf)
        results = load_standard_electrolyte_recipe(jf, exp_density=exp_density)
        if results:
            recipe_results.extend(results)
        else:
            logging.warning("文件 %s 未返回有效配方（已跳过）", jf)

    if not recipe_results:
        logging.error("没有可用的配方结果，退出。")
        raise SystemExit(1)

    # check smiles and molecular inventory
    mapping_smiles_list = {}
    for recipe_result in recipe_results:
        components = recipe_result['calculated_params']['components_calculated']
        for component in components:
            if component['component_type'] == 'salt':
                solute_map = auto_generate_solute_to_ions([component['name']])
                if component['name'] in solute_map:
                    cation_name, anion_name = solute_map[component['name']]
                    component['cation_name'] = cation_name
                    component['anion_name'] = anion_name
                    mapping_smiles_list[cation_name] = component.get('cation_smiles')
                    mapping_smiles_list[anion_name] = component.get('anion_smiles')
                else:
                    logging.warning("未为盐 %s 自动生成离子名映射，请检查。", component['name'])
            else:
                mapping_smiles_list[component['name']] = component['smiles']

    logging.info("映射 SMILES 列表：%s", list(mapping_smiles_list.keys()))

    # 优化后的匹配/映射
    mapped_smiles_dict, found_keys_dict = optimize_smiles_mapping(mapping_smiles_list, SMILES_DICT)
    logging.info("映射完成：未参数化数量=%d, 已存在数量=%d", len(mapped_smiles_dict), len(found_keys_dict))

    # create molecular params (保持原逻辑，但在发生错误时记录并跳过)
    try:
        from byteff2.train.utils import get_nb_params, load_model
        from byteff2.utils.mol_inventory import all_name_mapped_smiles
        from bytemol.core import Molecule
        from bytemol.utils import get_data_file_path
        import MDAnalysis as mda
        import ase.io as aio
    except Exception as e:
        logging.error("部分可选依赖未安装或加载失败：%s", e)
        get_nb_params = None
        load_model = None
        Molecule = None
        get_data_file_path = None
        mda = None
        aio = None

    OUTPUT_DIR = os.path.abspath("./params_results")

    def write_gro(mol, save_path: str):
        atoms_gro = mol.conformers[0].to_ase_atoms()
        atoms_gro.set_array('residuenames', np.array([mol.name] * mol.natoms))
        aio.write(save_path, atoms_gro)

    def gro2pdb_fix_box(gro_file: str, output_dir: str, mol, box_size: float = 60.0):
        box_nm = box_size / 10.0
        with open(gro_file, 'r') as f:
            lines = f.readlines()
        last_line = lines[-1].strip().split()
        if len(last_line) not in [3, 9]:
            lines[-1] = f"{box_nm:.5f} {box_nm:.5f} {box_nm:.5f}\n"
            with open(gro_file, 'w') as f:
                f.writelines(lines)

        pdb_file = f"{output_dir}/{mol.name}.pdb"
        u = mda.Universe(gro_file)
        for res in u.residues:
            res.resname = mol.name
        with mda.Writer(pdb_file, multiframe=False) as w:
            w.write(u.atoms)

    # 尝试加载模型（如果可用）
    model = None
    if get_data_file_path and load_model:
        try:
            model_dir = get_data_file_path('optimal.pt', 'byteff2.trained_models')
            model = load_model(os.path.dirname(model_dir))
        except Exception as e:
            logging.warning("加载模型失败：%s", e)
            model = None

    # 参数化未参数化分子
    for name, mps in mapped_smiles_dict.items():
        logging.info("处理分子：%s", name)
        OUTPUT_DIR_MOL = os.path.abspath(f"./template/input_data/{name}")
        if os.path.exists(OUTPUT_DIR_MOL):
            logging.info("已存在目录，跳过：%s", OUTPUT_DIR_MOL)
            continue
        try:
            if Molecule is None or get_nb_params is None or model is None:
                logging.warning("参数化工具未就绪，跳过分子参数化：%s", name)
                continue
            mol = Molecule.from_mapped_smiles(mps, nconfs=1)
            mol.name = name
            logging.info("调用 get_nb_params 生成参数：%s", name)
            if 'B' not in mps and 'Si' not in mps:
                metadata, params, tfs, mol = get_nb_params(model, mol)
                os.makedirs(OUTPUT_DIR_MOL, exist_ok=True)
                if os.path.exists(f'{OUTPUT_DIR_MOL}/{mol.name}.json'):
                    os.remove(f'{OUTPUT_DIR_MOL}/{mol.name}.json')
                tfs.write_itp(f'{OUTPUT_DIR_MOL}/{mol.name}.itp', separated_atp=True)
                write_gro(mol, f'{OUTPUT_DIR_MOL}/{mol.name}.gro')
                gro2pdb_fix_box(f'{OUTPUT_DIR_MOL}/{mol.name}.gro', OUTPUT_DIR_MOL, mol)
                with open(f'{OUTPUT_DIR_MOL}/{mol.name}.json', 'w') as f:
                    json.dump(params, f, indent=2)
                with open(f'{OUTPUT_DIR_MOL}/{mol.name}_nb_params.json', 'w') as file:
                    nb_params = {'metadata': metadata}
                    json.dump(nb_params, file, indent=2)
        except Exception as e:
            logging.error("处理分子 %s 失败：%s", name, e)
            traceback.print_exc()
            continue

    # create workdir and generate files
    os.makedirs(WORK_DIR, exist_ok=True)

    if not check_template_files(template_dir):
        logging.error("模板检查未通过，生成工作目录前请修复模板目录或传入正确 template_dir。")
        raise SystemExit(2)

    # 为每个配方生成工作目录与文件
    for recipe_result in recipe_results:
        logging.info('创建配方工作目录：%s', recipe_result['metadata']['recipe_name'])
        mass_fraction_list = recipe_result['calculated_params']['components_calculated']

        # 为盐生成 solute_to_ions 映射（若没有在 earlier step 写入）
        salt_names = [c['name'] for c in mass_fraction_list if c['component_type'] == 'salt']
        solute_to_ions_map = {}
        if salt_names:
            solute_to_ions_map = auto_generate_solute_to_ions(salt_names)

        try:
            counts = mass_fraction_to_counts(mass_fraction_list, total_molecules=1000, solute_to_ions=solute_to_ions_map)
        except Exception as e:
            logging.error("计算分子数失败：%s", e)
            continue

        recipe_result['calculated_params']['mol_counts'] = counts
        recipe_name = recipe_result['metadata']['recipe_name']
        folder = f"{WORK_DIR}/recipe_{recipe_name}"
        mol_counts = recipe_result['calculated_params']['mol_counts']
        temperature = recipe_result['simulation_params']['temperature']
        if os.path.exists(folder):
            logging.info("工作目录已存在，跳过：%s", folder)
            continue
        else:
            os.makedirs(folder, exist_ok=True)
            mol_masses = get_mol_masses(mol_counts, template_dir=template_dir)
            save_run_file(folder, mol_masses, mol_counts, temperature=temperature, template_dir=template_dir)
            save_pack_file(folder, mol_counts, template_dir=template_dir)
            save_topo_file(folder, mol_counts, template_dir=template_dir)
            logging.info("已生成工作目录并写入必要文件：%s", folder)

    logging.info("全部配方处理结束。输出目录：%s", WORK_DIR)
