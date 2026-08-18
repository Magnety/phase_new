# PHASE

PHASE（Phenotype-aligned, harmonized, adaptive self-supervised encoding）是一套放在本目录内的完整独立框架。DICOM/NIfTI/NPY 发现与预处理、患者级划分、缓存、增强、采样、模型、预训练、微调、推理、checkpoint、指标、中心审计、数据 QC、统计图和 3-D 模型解释均由 PHASE 自己实现；旧框架只作为结果基线，不参与任何 Python import 或 checkpoint 初始化。

默认数据索引也已固化为独立快照 `/home/ubuntu/liuyiyao1/multimodal_dataset/phase_source_manifest_full_v4.csv`；运行时不需要旧框架目录。

## 设计目标

旧实验的冻结测试结果为 pCR AUROC 0.612、HER2 AUROC 0.627；pCR 特征的标签探针 AUROC 只有 0.526±0.084，而中心探针 balanced accuracy 为 0.660±0.099（随机水平 0.200）。PHASE 针对这四类问题同时修改数据、预训练、模型和选择协议：

1. 微调和最终评估只选严格治疗前基线检查，但预训练使用所有病例的全部 DCE-capable 访视及其可用模态；预训练不读取临床标签，随后微调的训练/验证/测试患者划分仍严格只用基线访视，避免治疗后随访可用性进入下游分类。
2. 分层 3-D Transformer 以局部窗口保留乳腺形态，再在降采样 token 上建立全局空间关系；对应空间位置的跨时相 Transformer 与病例级连续时间 Transformer 共同提取局部和全局强化动力学。
3. 全新预训练同时优化双视图 VICReg、DCE/T1/T2/DWI/ADC token-MAE 体素重建、药代动力学 PINN、无位置编码泄漏的时相顺序恢复、缺失感知多模态 MoE、表型中心对抗、扫描风格中心分类和表型/风格正交。预训练会利用所有可用 DCE-capable 访视，不要求任何下游标签。
4. 第一层 MoE 只在病例实际存在的模态专家之间路由；第二层任务 MoE 用配置所选任务的嵌入激活共享专家。五个二分类端点进入二类原型头，分子分型进入四类原型头，生存端点进入 Cox 风险头。
5. 微调对同类别进行跨中心监督对比、类条件中心对齐和类条件域对抗，避免普通全局域对齐误删真实类别差异。
6. smooth GroupDRO 优化最差“任务×中心×类别”损失；采样同时考虑中心、任务、类别和缺失标签，但采用平方根逆频减少稀有病例反复记忆。
7. 分阶段解冻、小学习率骨干、标签平滑、训练集内类别权重、容量受限原型头和早停共同控制过拟合。
8. checkpoint 由验证集上各启用任务的全局与最差中心分数决定：二分类使用 AUROC、分子分型使用 macro OvR AUROC、生存使用 C-index。二分类阈值仅在验证集校准；测试集在模型选择完成前不构造，最终只读评估一次。

## 目录

实现按职责分包，顶层仅保留兼容入口：

```text
breast_dce_moe_pinn_phase/
├── cli/                 # 旧导入兼容层；实际 parse_arguments 在 main.py
├── configs/             # 预训练、微调、推理配置
├── data/                # manifest、患者划分、Dataset、采样器
├── preprocessing/       # 医学影像读取、变换、QC、发现和流水线
├── segmentation/        # 解剖 envelope、FTV proxy 与数据类型
├── models/              # 3-D/时间编码器、体素 MAE、两级 MoE、PINN、分类头
├── objectives/          # 预训练、分类和中心鲁棒目标
├── engine/              # 训练/推理编排与 checkpoint 原子读写
├── evaluation/          # 指标、审计、统计图和模型解释
├── main.py              # 旧命令行兼容入口（薄封装）
├── solver.py            # 旧 PHASESolver 导入兼容入口（薄封装）
└── losses.py            # 旧损失函数导入兼容入口（薄封装）
```

`data/`、`preprocessing/` 和 `segmentation/` 不依赖训练引擎；`models/` 不读取数据中心元数据；`objectives/` 只接收张量；`engine/` 负责把这些层组装成预训练、微调和只读评估。`evaluation/` 可由训练引擎调用，也可独立审计已有结果。公共命令行入口和输出组织保持不变。空间骨干升级后 checkpoint 格式为 `PHASE-independent-v5-spatiotemporal-transformer`；v4 CNN checkpoint 与新 token 拓扑不兼容，必须重新预训练。

框架外的回归测试位于 `tests/phase/`。`MODEL_DESIGN.md` 继续说明每个机制如何针对旧问题和提升分类准确率。

### 配置基准

三份默认配置的公共训练策略与 refine 基线对齐：输入为 `64×128×128`、每例最多 8 个 DCE 时相，预训练和微调均使用 AdamW、linear warmup + cosine 和 AMP。默认 v5 模型约 `25.06M` 参数；每个 DCE 体积先形成 `16×8×8=1024` 个细 token，经局部窗口注意力后合并为 `8×4×4=128` 个全局 token。它不是对 1024 个 token 直接做全局注意力的大型 ViT，容量和显存针对当前约 3405 个预训练访视及 1850 个可训练基线病例进行了约束。预训练默认全局 batch 为 72、mask ratio 为 0.60；三张 GPU 时微调默认全局 batch 为 24。`performance.parallel_mode` 可设为 `ddp`、`dp` 或 `single`，默认训练配置使用单进程 DP。DDP 一张卡一个进程，`batch_size` 是每个 rank 的大小；DP 只有一个进程，`batch_size` 是随后分到所有可见 GPU 的全局大小。DP 保持 compact case pack 在主机内，由三个持久线程把三个 shard 分别直接复制到目标 GPU，并用两个有界 host in-flight batches 准备后续 batch，避免先把完整 3-D batch 放到 GPU 0。预取会检查每张卡的空闲显存，低于配置阈值时自动只保留 CPU 准备并延迟到 scatter 时搬运，避免下一批抢占当前激活值。默认 `num_workers=0` 是有意的：PHASEDataset 已经在批内用 `sample_load_threads` 并行读取，3-D pack 再经过 DataLoader 多进程 IPC 通常会增加 `/dev/shm` 和序列化开销。DP deferred pack 不使用 DataLoader recursive pinning，以避免当前 Blackwell/PyTorch runtime 的 `CUDA error: invalid argument`；复制由各 GPU 独立 CUDA stream 并行完成。验证默认关闭 pin memory。DDP 仍可显式选择，但不会由默认配置启动。两种模式的双视图都在目标 GPU 内的一次并行 forward 中生成。预训练会持续导出 VICReg/MAE/PINN/域去偏/路由曲线、验证动力学拟合、表征与模态路由图；微调还会异步导出 ROC/PR、校准、决策曲线、bootstrap CI、中心/时间点亚组、MoE/PINN、表征探针和 3-D 模型解释。PHASE 默认输入模态为 `[DCE,T1,T2,DWI,ADC]`，非 DCE 模态全部允许缺失；MoE、PINN 与 token-MAE 均为当前独立实现的实际执行分支，不是仅存在于 YAML 的占位字段。

并行模式在 YAML 中选择；`--parallel-mode` 仅用于临时覆盖：

```yaml
performance:
  parallel_mode: dp  # ddp | dp | single
```

DP 要求 `--gpus` 至少提供两张卡，且全局 `batch_size` 不应小于卡数。RTX 5090/Blackwell 使用 DP 时还要求 NCCL `>=2.26.5`；NCCL `2.26.2` 在 PyTorch 2.7 + CUDA 12.8 的单进程多卡路径中可能触发 CUDA error 700，框架会在分配模型前拒绝该组合。可在训练使用的环境中执行 `/path/to/python -m pip install --no-deps --upgrade nvidia-nccl-cu12==2.26.5`，然后启动新的 Python 进程。DDP 不受此 DP 启动检查限制。`both` 和 `pipeline` 要求预训练与微调配置使用相同模式，因为同一次运行不能在中途改变进程拓扑。

`main.py` 直接提供与 refine 对齐的 `parse_arguments`，支持 `--pretrain-config`、`--finetune-config`、`--infer-config`、`--split-mode`、数据中心筛选、比例和 DataLoader 覆盖；默认 seed 为 2026。默认 `by_ratio` 先以 refine 相同的 `SHA1(seed|dataset_id|patient_id)` 规则锁定测试患者（0.15），随后仅对余下患者按中心×pCR/未标注状态重分训练（0.70）与验证（0.15），因此 PHASE 的分层调整不会改变 refine 已固定的测试集。

### 任务选择与缺失标签

用户列出的项目按端点计实际是 7 个（生存时间和事件共同组成一个生存端点）：`pCR、HER2、ER、PR、HR、molecular_subtype、survival`。三份配置均可选择任意非空子集：

```yaml
tasks:
  active: [pCR, HER2, ER, PR, HR, molecular_subtype, survival]
  molecular_subtype_classes: ["0", "1", "2", "3"]
```

五个受体/疗效标签按二分类读取；分子分型按配置类名映射；生存只有在 `survival_time>0` 且 `survival_event∈{0,1}` 时有效，其中 `event=0` 是右删失而不是负类。每个任务都有独立 `label_mask`：某任务缺失只跳过该任务的损失、分层组和指标，不会删除病例，也不会屏蔽同一病例的其他已知标签。当前清单中可训练基线病例为 1850，pCR/HER2/ER/PR/HR/分型/生存有效标签数分别为 1752/1534/891/891/1537/765/260；260 个生存标签中有 21 个事件和 239 个删失。

## FTV 不再聚焦腹部

原先的“全图强化阈值 + 最大连通域/最亮体素”并不是真正的乳腺分割：心脏、肝脏、腹部血管或线圈边缘强化更强时会直接胜出，提高强化阈值反而会加重这个问题。PHASE 现在使用一条完全独立、病例内推断的安全链路：

1. 从 pre-contrast 图像估计身体前景；阈值同时参考图像边界背景，避免 Otsu 只留下明亮心脏而漏掉低信号乳房。
2. 在每个轴位切片计算身体表面深度，排除深部胸腹器官和最外层皮肤/边缘伪影；不假定乳房固定朝上、下、左或右。
3. 利用身体前景质心偏向体积更大的胸腹侧这一几何事实，在病例内推断相反的乳房侧，并把轴向两端的腹部高风险切片排除。该方向和置信度写入 QC，但中心 ID 从不参与推断或模型前向。
4. 只在乳腺 envelope 内形成增强候选；按体积上限、三轴跨度、包围盒填充率、体积边界接触、早期增强和晚期持续性评分。没有合格连通域时 `ftv_mask_valid=false`，不会回退到全图最亮的 16/32 个体素。
5. 模型空间注意力被乳腺 envelope 硬约束，envelope 外信号衰减到 0.10；有效 FTV 只增加 soft attention bias，不做硬裁剪。动力学曲线同样先乘乳腺 mask，再对有效 FTV 作软加权。因此 FTV 无效时仍不会重新关注腹部，同时避免错误 FTV 完全控制分类。
6. 离线预处理保存 `anatomical_breast_mask.npy`、`ftv_mask.npy`，并在 manifest、`preprocessing.json`、`quality_control.csv` 和预览图中记录有效性、比例、拒绝原因、推断方向和轮廓。训练增强会对图像和两个 mask 做完全相同的翻转/旋转。

在 9 个数据中心各取 5 个真实病例的回归抽查中，全图最亮强化体素有 34/45 位于乳腺 envelope 外；新 FTV 有效 35/45，另外 10 例失败关闭，并且 45 例中没有一例把全图最亮点收入 FTV。所有有效 FTV 的轴向质心均位于体积的 0.238–0.806，未落入配置排除的腹部端区。该检查验证的是定位安全性，不替代肿瘤标注 Dice 或最终 pCR/HER2 冻结测试。

## 运行

从原始 manifest 完成预处理、数据可视化、预训练和微调的全流程：

```bash
/home/ubuntu/demo/.conda/pkgs/torchnew/bin/python3.10 \
  -m src.breast_mri_ai.breast_dce_moe_pinn_phase \
  --mode pipeline --config src/breast_mri_ai/breast_dce_moe_pinn_phase/configs/finetune_phase.yaml \
  --gpus 1,2,3
```

`pipeline` 的离线预处理按病例保存完成标记，断电重启会跳过版本一致且文件齐全的病例。

如果已有合格的 NPY manifest，可直接从头完成预训练和微调：

```bash
/home/ubuntu/demo/.conda/pkgs/torchnew/bin/python3.10 \
  -m src.breast_mri_ai.breast_dce_moe_pinn_phase \
  --mode both --config src/breast_mri_ai/breast_dce_moe_pinn_phase/configs/finetune_phase.yaml \
  --gpus 1,2,3
```

只运行预训练：

```bash
/home/ubuntu/demo/.conda/pkgs/torchnew/bin/python3.10 \
  -m src.breast_mri_ai.breast_dce_moe_pinn_phase \
  --mode pretrain --gpus 1,2,3
```

只做离线预处理或数据可视化：

```bash
/home/ubuntu/demo/.conda/pkgs/torchnew/bin/python3.10 \
  -m src.breast_mri_ai.breast_dce_moe_pinn_phase --mode preprocess

/home/ubuntu/demo/.conda/pkgs/torchnew/bin/python3.10 \
  -m src.breast_mri_ai.breast_dce_moe_pinn_phase --mode visualize-data
```

从没有 manifest 的原始 DICOM 目录生成可人工复核的候选 manifest：

```bash
/home/ubuntu/demo/.conda/pkgs/torchnew/bin/python3.10 \
  -m src.breast_mri_ai.breast_dce_moe_pinn_phase.preprocessing \
  discover-dicom --raw-root /path/to/raw --dataset-id new_center \
  --labels-csv /path/to/labels.csv --output-manifest /path/to/source_manifest.csv
```

使用 PHASE 预训练 checkpoint 微调：

```bash
/home/ubuntu/demo/.conda/pkgs/torchnew/bin/python3.10 \
  -m src.breast_mri_ai.breast_dce_moe_pinn_phase \
  --mode finetune --gpus 1,2,3 \
  --checkpoint /path/to/best_pretrain.pth
```

断点恢复时，`--resume` 必须匹配当前阶段；推理必须使用微调 checkpoint：

```bash
/home/ubuntu/demo/.conda/pkgs/torchnew/bin/python3.10 \
  -m src.breast_mri_ai.breast_dce_moe_pinn_phase \
  --mode infer --gpus 1,2,3 \
  --checkpoint /path/to/best_finetune.pth
```

结果目录包含 `summary.json`、`history.json`、`calibration.json`、checkpoint、逐病例预测 CSV、所有任务特征 NPZ、表示探针 JSON和统计图。二分类导出标签/概率及 ROC/PR；分子分型导出真实/预测类别、每类概率和混淆矩阵；生存导出时间、事件、log-risk、C-index 和风险图。逐病例 CSV 还记录模态可用性/路由权重、所有任务专家权重、PINN 的 Ktrans/ve/vp/kep/BAT、拟合置信度和曲线 RMSE。

## 正式验收

代码结构不能保证未经运行的数据集指标。是否真正解决问题必须以冻结测试和多种子/留一中心实验判断。预先固定的核心判据是：pCR、HER2 的 AUROC 与 accuracy 均高于旧基线；最差中心 AUROC 上升；中心探针下降而任务探针上升；训练—验证差距和 Brier 不再持续恶化。不要根据训练集图或单一随机种子宣称成功。
