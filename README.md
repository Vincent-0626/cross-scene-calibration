# 面向跨场景图像分类的准确率与置信度研究 (Accuracy and Confidence in Cross-Scene Image Classification)

本仓库包含论文 **《面向跨场景图像分类的准确率与置信度研究》** 的完整实验训练脚本、后处理校准代码、数据划分清单与 SHA-256 校验、模型预测 logits (`logits.npz`)、统计分析结果 (`analysis/`) 以及论文图表复现脚本。

---

## 1. 仓库目录结构 (Repository Structure)

```text
├── analysis/                                  # 由保存的 logits 无损重算的全部汇总指标与统计检验
│   ├── all_run_metrics.csv                    # 全部主实验与多随机种子运行的完整指标汇总
│   ├── three_seed_summary.csv                 # 三颗随机种子 (42, 7, 2026) 均值 ± 样本标准差汇总 (表3.6)
│   ├── reliability_bins.csv                   # 10 分箱置信度与经验准确率明细 (图3.3)
│   ├── plant_23_vs_28.csv                     # PlantDoc 28 标准类与 23 严格对齐类敏感性对比 (表3.5)
│   ├── plant_cluster_bootstrap.csv            # 按 236 张原图整图有放回重采样 2000 次的 95% 置信区间
│   └── plant_paired_differences.csv           # 相对基线模型的配对差异 Bootstrap 区间
├── data/                                      # 服务器端全部实验脚本、配置、划分清单与推理结果
│   ├── calibration_three_domains/             # 基础校准对比 (温度缩放、标签平滑、置信度惩罚、平滑+温度)
│   │   ├── experiment.py                      # 三领域统一校准实验主脚本
│   │   ├── collect_results.py                 # 指标汇总脚本
│   │   ├── comparison.csv                     # 基础校准对比汇总表 (表3.1, 表3.3)
│   │   └── {plant,office,visda}/              # 各领域各方法的 result.json, config.json, logits.npz, epochs.csv
│   ├── dac_augmix/                            # 进阶训练与后处理校准实验 (AugMix, DAC, 平滑+熵, ALiS)
│   │   ├── augmix_experiment.py               # AugMix 一致性训练脚本
│   │   ├── dac_experiment.py                  # 密度感知校准 (DAC) 特征提取与近邻温度拟合脚本
│   │   ├── ls_entropy_experiment.py           # 标签平滑 + 熵正则化训练脚本
│   │   ├── alis_experiment.py                 # ALiS 自适应平滑与联合温度缩放脚本
│   │   ├── run_improvements.py                # T>=1 非锐化约束与“强分类骨干+DAC”协同改进实验脚本 (表3.4)
│   │   ├── improvements_20261124/results.json # 协同改进实验完整输出 (表3.4)
│   │   ├── seed_repeats/                      # 随机种子 7 与 2026 重复训练脚本与完整输出 (表3.6, 图3.5)
│   │   ├── office_multisource/                # Office-Home 多源域 (Product+Art+Clipart -> Real World) 与 MixStyle 实验
│   │   ├── counterfactual_leaf.py             # PlantDoc GrabCut 前景分割与背景替换对照脚本
│   │   └── official_dac/                      # DAC 核心优化器与 k-NN 距离计算模块
│   ├── target_domain_baselines_20261122/      # 目标域同场景训练参考基准 (图3.1)
│   ├── plantdoc_experiment/                   # PlantVillage / PlantDoc 基础训练脚本与标注框清单 (artifacts/*.json.gz)
│   ├── visda2017/                             # VisDA-2017 源域训练脚本与数据划分
│   └── metadata/                              # 三领域样本图与类别映射清单
├── figures/                                   # 论文核心图表输出 (300 DPI)
│   ├── fig_2_1_domain_samples.png             # 图2.1 三领域源域与目标域样本对比
│   ├── fig_3_3_reliability.png                # 图3.3 三领域代表性方法可靠性图
│   └── fig_3_4_three_seeds.png                # 图3.5 三颗随机种子准确率与 ECE 均值±标准差
├── work/                                      # 本地一键复现分析与绘图脚本 (无需 GPU)
│   ├── analyze_predictions.py                 # 从 data/**/logits.npz 严格校验并重算全部论文表格指标
│   └── make_figures.py                        # 一键生成 figures/ 下的论文图表
└── requirements.txt                           # Python 依赖环境
```

---

## 2. 快速复现论文指标与图表（无需 GPU / 无需下载原始图像）

本仓库已在 `data/**/logits.npz` 中保存了各领域所有方法在源域验证集与目标域测试集上的完整输出 logits 及真实标签。只需安装基础科学计算依赖，即可在数秒内精确复现论文中的全部表格数值、Bootstrap 置信区间与图表：

```bash
pip install -r requirements.txt

# 1. 校验全部运行结果并重新生成 analysis/ 下的所有 CSV/JSON 统计表
python work/analyze_predictions.py --require-three-seeds --bootstrap 2000

# 2. 重新生成 figures/ 下的论文图表
python work/make_figures.py
```

`work/analyze_predictions.py` 内置了数值一致性断言（复算准确率、ECE、NLL 与服务器原始 `result.json` 的误差严格小于 `5e-7`）。

---

## 3. 从原始数据集重新训练与推理（GPU 服务器复现指南）

### 3.1 数据集准备
1. **PlantVillage → PlantDoc (`plant`)**：
   - 源域：`PlantVillage` 彩色叶片图像（28 个映射类别，26,979 训练 / 5,781 验证）。
   - 目标域：`PlantDoc` 测试集标注叶片框（452 个测试框，源自 236 张原图；精确坐标与 SHA-256 哈希见 `data/plantdoc_experiment/artifacts/leaves.json.gz` 与 `config.json.gz`）。
2. **Office-Home (`office`)**：
   - 下载 [Office-Home Dataset](https://www.hemanthdv.org/officeHomeDataset.html)，使用 `Product`（3,538 训练 / 885 验证）为源域、`Real World`（4,327 测试）为目标域，划分清单见 `data/runs/office_product_to_real/{train,val,test}.json`。
3. **VisDA-2017 (`visda`)**：
   - 下载 [VisDA-2017 Classification](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)，源域合成图（121,837 训练 / 30,560 验证）与目标域真实图（55,388 测试）划分清单见 `data/visda2017/runs/source_only_30/splits.json`。

### 3.2 运行各阶段实验
在配置好数据集路径的 GPU 环境（论文实验使用单张 NVIDIA GeForce RTX 4090，PyTorch + ImageNet 预训练 `ResNet-50`）中依次执行：

```bash
# 1. 基础方法与后处理校准 (原模型、温度缩放、标签平滑、置信度惩罚)
for domain in plant office visda; do
  for method in temperature label_smoothing confidence_penalty; do
    python data/calibration_three_domains/experiment.py --domain $domain --method $method
  done
done

# 2. 进阶方法 (AugMix, DAC, 标签平滑+熵正则化, ALiS)
for domain in plant office visda; do
  python data/dac_augmix/augmix_experiment.py --domain $domain --method augmix
  python data/dac_augmix/dac_experiment.py --domain $domain
  python data/dac_augmix/ls_entropy_experiment.py --domain $domain
  python data/dac_augmix/alis_experiment.py --domain $domain --method alis
done

# 3. 多随机种子 (seed=7, 2026) 重复训练实验
bash data/dac_augmix/seed_repeats/queue.sh

# 4. 机理改进实验：T>=1 非锐化约束与“强分类骨干 + DAC”协同校准 (表3.4)
python data/dac_augmix/run_improvements.py
```

---

## 4. 核心实验结论速览

1. **三颗随机种子（`42, 7, 2026`）主结果（均值 ± 标准差）**：
   - **植物病害（PlantDoc）**：`平滑＋熵` 准确率最高（**32.08±0.80%** vs 原模型 26.40±4.02%），ECE 降至 **25.96±3.96%**（原模型 52.78±2.45%）。
   - **日常物品（Office-Home）**：`平滑＋熵` 准确率最高（**66.10±1.64%** vs 原模型 62.11±0.11%）；`ALiS` 校准最稳（ECE **5.32±0.28%**，准确率 64.39±0.45%）。
   - **合成到真实（VisDA）**：`AugMix` 与 `ALiS` 将平均准确率从原模型的 17.79±3.94% 提升至 **23.98±4.01%** 与 **25.46±3.95%**。
2. **后处理温度缩放 $T<1$ 反向锐化机理与协同改进（`seed=42`）**：
   - 固定标签平滑在源域制造约 8.65–10.11 个百分点的欠自信，诱导源域 NLL 拟合出 $T<1$ 的锐化温度（`0.5330–0.7305`），在植物与 VisDA 目标域引发严重的过度自信恶化。
   - 施加 **$T \ge 1$ 非锐化约束**可彻底消除反向锐化；将各领域最优训练模型与 **DAC** 结合后，植物（`平滑＋熵＋DAC`）、日常物品（`平滑＋熵＋DAC`）与 VisDA（`AugMix＋DAC`）在不损失准确率的前提下将目标域 ECE 分别压低至 **24.18%**、**4.60%** 与 **23.80%**。
