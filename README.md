# ZX-GBS+QUBO：GBS 量子特征增强的投资组合优化

基于高斯玻色采样（GBS）特征增强的量化投资组合优化研究项目。核心链路：金融因子
→ GBS 采样特征（仿真 / 真机）→ 读出模型 → 组合优化 → 滚动回测。

## 环境要求

- Python ≥ 3.11
- macOS / Linux

## 最小启动流程

### 1. 创建虚拟环境并安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

可选依赖（按需安装）：

```bash
pip install -e ".[gbm]"      # C3 实验需要 lightgbm
pip install -e ".[gbs-sim]"  # 精确 GBS 仿真（strawberryfields/thewalrus），一般不需要
```

### 2. 验证安装（约 2 分钟）

```bash
pytest -q
# 预期：249 passed
```

### 3. 跑通第一个实验（mock 数据，无需任何外部数据）

```bash
PYTHONPATH=src python -c "
from qgbs.experiments.runner import run_experiment
r = run_experiment('configs/experiments/c1_classical_baseline.yaml')
print(r.experiment_id, r.metrics)
"
```

几秒内输出 C1 经典基线的回测指标（sharpe、rank_ic、max_drawdown 等）。

### 4. 批量执行对照矩阵（可选）

```bash
PYTHONPATH=src python -c "
from qgbs.experiments.runner import run_all
print(run_all())                        # 默认 C1-C5（mock）
print(run_all(['Q1', 'Q2', 'Q3', 'Q4'])) # GBS 特征族（仿真样本）
"
```

## 接入真实数据（中信交付）

仓库不含真实数据（`.gitignore` 已排除）。拿到交付 CSV 后：

```bash
# 1. 将交付文件放入 data/（静态数据_1.csv / 行情数据_1.csv / 指数相关时序数据_1.csv）
# 2. ETL 生成标准化数据仓
PYTHONPATH=src python -m qgbs.data.etl_zx --data-dir data --out-dir data_store

# 3. 运行真实数据实验（*_csv.yaml 后缀的配置）
PYTHONPATH=src python -c "
from qgbs.experiments.runner import run_experiment
r = run_experiment('configs/experiments/c1_classical_baseline_csv.yaml')
print(r.metrics)
"

# 4. P2 诊断（置换对照 / 低阶统计匹配对照 / 批次稳定性）
PYTHONPATH=src python scripts/p2_diagnostics_csv.py
# 产物输出至 artifacts/p2_csv/
```

## 目录结构

```
configs/            实验配置（base.yaml + experiments/ 下 C0-C5、Q1-Q4，mock 与 csv 双版本）
src/qgbs/           核心代码
  ├── data/         数据适配器（mock / csv）与 ETL
  ├── factors/      因子构建、中性化、流水线
  ├── gbs/          GBS 采样后端（simulator / cached / hardware）与样本库
  ├── models/       读出模型
  ├── portfolio/    组合优化（cvxpy）
  ├── backtest/     滚动回测引擎与指标
  └── experiments/  配置驱动的实验执行器（runner.py 为入口）
scripts/            辅助脚本（P2 诊断等）
tests/              pytest 测试套件
docs/               方案与阶段报告
data_store/         ETL 产物（真实数据，不入库）
artifacts/          实验产物（不入库）
```

## 实验编号约定

- **C0-C5**：经典对照（基线复现、修正基线、Ridge、GBM、分位数显式、随机特征）
- **Q1-Q4**：GBS 特征族（边际、二阶、高阶、全量增强）
- 配置同名双版本：无后缀 = mock 数据；`_csv` 后缀 = 中信真实交付数据
- `_hardware` 后缀 = 真机 GBS 样本（批次入库见 `scripts/ingest_hardware_samples.py`；
  按路径运行，不注册进 registry，如 `q1_marginal_hardware.yaml`）

## 注意事项

- 仿真 GBS 样本为近似口径（高斯 copula 阈值模型），用于管线验证，**不替代真机结论**。
- `data_store/constraints_costs.csv` 当前为占位值，禁止用于结论性产出。
- 资产池 zz800 为 2025 年末快照（非 point-in-time），仅限方法学验证。
