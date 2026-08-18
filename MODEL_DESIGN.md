# PHASE v5：分层 3-D 时空 Transformer 基础模型

## 1. 总体架构

PHASE v5 将原来的 CNN 空间骨干替换为面向小中规模医学影像数据的分层 Transformer，同时保留多模态 MoE、药代 PINN、异构任务头和中心鲁棒训练：

```text
每个 DCE 时相 [1,64,128,128]
  -> 3-D PatchEmbed, patch=[4,16,16]
  -> 细网格 [192,16,8,8]，1024 tokens
  -> 2 层局部窗口 Transformer, window=[4,4,4]
  -> 2x2x2 learned patch merge
  -> 粗网格 [384,8,4,4]，128 tokens
  -> 2 层全局空间 Transformer
  -> 对应空间位置的 2 层跨时相 Transformer
  -> anatomy/FTV 引导的空间注意力池化
  -> 每时相 384 维向量
  -> 2 层连续时间病例级 Transformer
  -> first / mean / delta / kinetic 融合
  -> 128 维 DCE phenotype

T1/T2/DWI/ADC
  -> 各自独立的浅层分层 3-D Transformer
  -> 128 维模态特征

DCE + 可用辅助模态 -> 缺失感知模态 MoE
DCE ROI 曲线 + 采集时间 -> extended-Tofts PINN
模态表型 + PINN -> 任务条件 MoE
  -> pCR/HER2/ER/PR/HR 二类原型头
  -> molecular_subtype 四类原型头
  -> survival Cox 风险头
```

中心 ID 不进入编码器、时相模块、路由器、PINN 或预测头，只用于训练中的域对抗和审计。

## 2. 为什么不是直接使用大型 3-D ViT

默认模型约 25.06M 参数，其中 DCE 空间编码器 4.81M、patch-wise 时相编码器 2.52M、病例级时相编码器 3.55M、四个辅助空间编码器共 11.86M。当前约有 3405 个可用于无标签预训练的 DCE-capable 访视和 1850 个可训练基线病例，因此没有采用所有细 token 全局互注意的大型 ViT。

细粒度阶段只在 `4x4x4=64` token 的局部窗口中做注意力，优先学习边缘、肿块形态和局部组织关系；合并到 128 token 后才做全局注意力，用于双侧乳腺和远距离背景关系。辅助模态只有 1 层局部和 1 层全局块，DCE 主干才使用 2+2 层。该设计在保留 Transformer 的可扩展性与 MAE 兼容性的同时，限制参数量、二次注意力成本和过拟合风险。

## 3. 时相输入与两级时间建模

输入最多 8 个 DCE 时相，使用 `phase_mask` 屏蔽缺失时相，使用归一化连续位置和分钟制真实采集时间，不要求每个中心具有完全相同的相数或时间间隔。

第一级时间建模在粗空间网格的对应位置进行。张量由 `[B,P,C,8,4,4]` 重排为 `[B*128,P,C]`，每个乳腺位置独立观察多时相强化轨迹，因此局部 wash-in、plateau 和 wash-out 不会在全局池化前被抹平。缺失时相作为 key padding mask；分块处理空间位置以控制峰值显存。

第二级时间建模作用于池化后的每时相向量，使用连续时相嵌入建模全乳腺动力学。最终融合首时相、有效时相均值、末首差和原始 ROI 动力学摘要。这样既有位置特异的局部变化，也有病例级全局变化。

## 4. 输入与输出

默认输入为：

- `volumes`: DCE，`[B,P,1,64,128,128]`，`P<=8`；
- `phase_mask`: 各时相是否真实存在；
- `phase_positions`: 归一化连续采集位置；
- `phase_times`: 分钟制采集时间，供 PINN 使用；
- `auxiliary_volumes`: 固定槽位 T1/T2/DWI/ADC，均可缺失；
- `modality_mask`: 每个模态是否真实采集；
- `anatomical_mask` 与可选 `ftv_mask`: 限制乳腺区域并软引导病灶注意力。

微调输出包含各启用任务的 logits/risk、任务特征、两级 MoE 路由权重、空间注意力、药代参数 `Ktrans/ve/vp/kep/BAT` 及拟合曲线。预训练额外输出双视图投影、各模态重建、voxel mask、时相顺序 logits 和域分支结果。

## 5. Transformer token-MAE 预训练

MAE 不再把原始体素改成固定 mask value。流程为：

1. 在 `[8,16,16]` 的体素块网格采样 60% mask；
2. 输入仍经过 3-D PatchEmbed，被遮挡位置的投影 token 替换为可学习 mask token；
3. 未遮挡 token 经过局部和全局空间 Transformer；
4. DCE 粗 token 再经过对应位置跨时相 Transformer，使其他时相能为当前遮挡区域提供动力学上下文；
5. 解码器融合细空间特征和粗时空上下文，重建原始尺寸体积；
6. Smooth-L1 只在被遮挡体素计算，乳腺 envelope 外权重降低为 0.1。

完整无标签目标还包括双视图 VICReg、所有可用辅助模态 MAE、PINN 曲线/ODE、时相顺序、模态路由正则、中心对抗、style 中心分类和 phenotype/style 正交。时相顺序头从任何连续位置嵌入和时间 Transformer 之前的 image-only 向量分叉，不能通过读取位置编码完成任务。

## 6. 微调策略

微调必须使用 v5 预训练 checkpoint，除非显式开启随机初始化消融。默认分阶段训练：

1. head warmup：冻结 DCE 空间骨干、两级时相骨干、辅助模态骨干、模态 MoE 和 PINN，先训练任务 MoE、adapter 与任务头；
2. 中间阶段：保持空间编码器冻结，解冻 patch-wise 和病例级时相模块，让任务先适配强化轨迹；
3. `spatial_unfreeze_epoch` 后：以骨干学习率倍率逐步联合微调空间 Transformer；
4. `classification_detach_pinn=true` 时持续冻结 PINN 本体，仅训练其临床适配器，避免分类梯度破坏药代参数。

监督目标为缺失标签安全的多任务损失，并加入跨中心监督对比、类条件中心对齐、类条件域对抗、GroupDRO、原型分离和两级 MoE 正则。测试患者在模型选择前不参与阈值或 checkpoint 选择。

## 7. 兼容性与结论边界

checkpoint 格式为 `PHASE-independent-v5-spatiotemporal-transformer`。v4 CNN 权重的参数拓扑与 token 表示均不兼容，加载时会明确拒绝，必须重新预训练。

分层局部/全局注意力、双层时相建模和 25M 容量是针对当前数据规模作出的架构约束，不等于已经证明临床指标提升。最终仍需以冻结测试、多随机种子和留一中心实验验证 pCR/HER2、最差中心性能、校准误差及任务/中心探针。
