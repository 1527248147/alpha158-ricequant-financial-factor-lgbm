# A-Share Stock Selection with 400+ Factors + LightGBM (Alpha158 + Ricequant)

基于 **400+ 特征（因子）与机器学习** 的 A 股日频选股策略。  
核心结果：**Rank IC ≈ 0.07**，**2025 样本外净收益 ≈ +30%**（TopK=50，含保守交易约束）。

---

## Project Overview

### 数据工程（2005-01-01 ~ 2025-12-15）
- 整合两类特征源：
  - **Alpha158**：基于 OHLCV 的衍生技术因子
  - **Ricequant 财报/基本面日频因子（300+）**
- 构建覆盖 **2005-01-01 ~ 2025-12-15** 的 A 股日频面板数据（交易日 × 股票）
- 实现：交易日历对齐、异常值/缺失处理、截面标准化与特征一致性校验
- 产出可复用的高质量训练/回测数据集（面向 ML pipeline）

### 机器学习建模
- **严格标签（Strict Label）**：  
  \[
  y_t = \frac{Close(t+2)}{Close(t+1)} - 1
  \]
- 使用 **LightGBM** 进行回归/排序信号学习
- 按日输出全市场预测分数，用于截面选股

### 特征筛选与稳定性
- 标准化因子筛选流程：缺失率/稳定性过滤 + LGBM 正则化/重要性约束
- 将 ~400 因子自动收敛到“有效因子子集”，提升泛化并降低噪声维度

### 回测与交易约束
- TopK 组合策略（TopK=50）
- 纳入保守交易约束：
  - 手续费
  - 涨跌停/停牌不可交易
  - IPO 冷静期 / 新股过滤
  - T+1 资金结算等
- 2025-01-02~2025-12-10：
  - 净收益（含手续费）：**≈ +30%**
  - 零成本收益：**≈ +39%**
  - Rank IC mean：**≈ 0.0696**
  - IC IR：**≈ 0.6063**
    
 ### 最终回测曲线
 ![2025 NAV: Strategy vs CSI 300](project_alpha158+ricequant_fin+lgbm/strategy_vs_csi300_2025.png)
---

## Dataset Download (Baidu Netdisk)

本仓库不包含完整 `dataset/`（体积较大）。请通过百度网盘下载后放入项目工作目录。

通过网盘分享的文件：alpha158+ricequant_finance+lgbm数据集
链接: https://pan.baidu.com/s/1yIGTYrIe21nmIMGytfFmeg?pwd=6h8a 提取码: 6h8a 
--来自百度网盘超级会员v4的分享

下载完成后，请确保目录结构如下（重点是 `dataset` 放在 `project_alpha158+ricequant_fin+lgbm` 下）：

```
project_alpha158+ricequant_fin+lgbm/
  dataset/
    rq_ohlcv_yearly_parquet/
    trading_calendar_from_merged.csv
  labeled/
  train_models/
  backtest_20250101_20251210/
```

---

## How to Run (Windows CMD)

以下命令为 Windows CMD（使用 `^` 换行续写）。

### Step 1 — 打标签（Labeling）
```bat
cd /d C:\AI_STOCK\project_alpha158+ricequant_fin+lgbm

python labeled\build_labels_strict_by_calendar_from_ohlcv.py ^
  --feature_dir "labeled" ^
  --ohlcv_dir   "dataset\rq_ohlcv_yearly_parquet" ^
  --calendar_csv "dataset\trading_calendar_from_merged.csv" ^
  --out_dir "_calendar" ^
  --price_col close ^
  --mode append_full
```

标签生成后，特征与标签将位于：
```
labeled\_calendar\labeled_yearly_parquet
```

---

### Step 2 — 训练 LightGBM 模型（Training）
```bat
cd /d C:\AI_STOCK\project_alpha158+ricequant_fin+lgbm

python train_models\train_lgbm_lambdarank_strict_calendar_v4_fundselect_validperm.py ^
  --data_dir "labeled\_calendar\labeled_yearly_parquet" ^
  --out_dir  "train_models\_train_lambdarank_v4_alpha_plus_fundTop15_seed42" ^
  --train_years 2021-2023 --valid_years 2024 --test_years 2025 ^
  --label_col label__ret_1d_qlib ^
  --missing_drop_thresh 0.98 ^
  --cast_float32 ^
  --exclude_regex "(?i)dividend" ^
  --fund_perm_csv "train_models\feature_importance_valid_perm.csv" ^
  --fund_perm_topk 15 ^
  --fund_perm_min_drop 1e-4 ^
  --clip_y_abs 0 ^
  --relevance_bins 10 ^
  --min_group_size 30 ^
  --truncation_level 50 ^
  --num_boost_round 5000 ^
  --early_stopping_rounds 400 ^
  --save_test_pred ^
  --seed 42 ^
  --fund_perm_min_drop 0
```

训练完成后，模型与结果输出在：
```
train_models\_train_lambdarank_v4_alpha_plus_fundTop15_seed42
```

---

### Step 3 — 复现回测（Backtest）
```bat
cd /d C:\AI_STOCK\project_alpha158+ricequant_fin+lgbm

python backtest_20250101_20251210\backtest_topk_dropout.py ^
  --model_dir "train_models\_train_lambdarank_v4_alpha_plus_fundTop15_seed42" ^
  --features_dir "labeled\_calendar\labeled_yearly_parquet" ^
  --ohlcv_dir "dataset\rq_ohlcv_yearly_parquet" ^
  --calendar_csv "dataset\trading_calendar_from_merged.csv" ^
  --out_dir "backtest_20250101_20251210\out_topk_dropout_2025" ^
  --topk 50 --n_drop 5 --init_cash 1000000 --settlement_t1
```

回测输出在：
```
backtest_20250101_20251210\out_topk_dropout_2025
```

---

## Repo Structure

- `project_alpha158+ricequant_fin+lgbm/`
  - `dataset/`（网盘下载后放入；大文件不提交 Git）
  - `labeled/`（标签构建与严格交易日历对齐）
  - `train_models/`（LightGBM 训练与特征筛选）
  - `backtest_20250101_20251210/`（回测脚本与回测输出）

---

## Disclaimer

本项目仅用于研究与学习交流，不构成任何投资建议。历史回测不代表未来表现。
