# SMILES to OpenMM ForceField XML + PDB

这个示例目录用于把一个分子的 SMILES 转成：
- 可由 `openmm.app.ForceField` 直接加载的 `*_ff.xml`
- 对应的 `*.pdb`

并提供能量验证脚本，检查 XML 加载后的体系能量是否可正常计算。

## 核心脚本

- `generate_ff_xml_pdb.py`：从 SMILES 生成 `*_ff.xml` 与 `*.pdb`
- `load_ff.py`：注册并加载自定义 `<ByteFF14Force>` XML 标签
- `test_energy.py`：用生成的 XML/PDB 创建 OpenMM System 并输出总能量与分项能量
- `test_energy_old.py`：旧流程对照脚本（AmoebaCalculator vs ForceField XML）

## 快速开始

### 1) 生成乙醇文件

```bash
python generate_ff_xml_pdb.py --smiles "CCO" --name ethanol --output ethanol_files
```

默认会在 `ethanol_files/` 下生成：
- `ethanol_ff.xml`
- `ethanol.pdb`
- `params/`（中间参数文件）
- `working/`（构建过程文件）

### 2) 计算能量（新流程）

```bash
python test_energy.py --xml ethanol_files/ethanol_ff.xml --pdb ethanol_files/ethanol.pdb
```

可选平台参数：

```bash
python test_energy.py --xml ethanol_files/ethanol_ff.xml --pdb ethanol_files/ethanol.pdb --platform CUDA
```

`--platform` 支持：`CPU`、`Reference`、`CUDA`。

## 多分子/多 XML 场景

`load_ff.py` 支持同时加载多个 `*_ff.xml` 文件（原子类型通过哈希保证全局唯一）：

```python
from load_ff import load_forcefield
ff = load_forcefield("mol1_files/mol1_ff.xml", "mol2_files/mol2_ff.xml")
```

然后使用包含对应残基模板的 PDB 调用 `ff.createSystem(...)`。

## 与旧流程对照（可选）

如果想对比旧方法（直接从 Gromacs 参数经 `AmoebaCalculator`）和新方法（`app.ForceField` + XML），可运行：

```bash
python test_energy_old.py \
  --top ethanol_files/params/system_gas.top \
  --json ethanol_files/params/ethanol.json \
  --nbparams ethanol_files/params/ethanol_nb_params.json \
  --name ethanol \
  --pdb ethanol_files/ethanol.pdb \
  --xml ethanol_files/ethanol_ff.xml
```

## 说明

- 本目录生成的是 **ForceField XML**（`*_ff.xml`），不是 `XmlSerializer.serialize(system)` 的 serialized System XML。
- 新流程依赖 `load_ff.py` 注册 `<ByteFF14Force>` 解析器；直接 `app.ForceField(...)` 读取该 XML 前，需要先导入或调用 `load_forcefield(...)`。
