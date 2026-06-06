# SSMH 与现有 AAAI 工作对比及基线规划

更新时间：2026-06-06

## 0. 对比范围与结论

本文中的“现在的 AAAI”按两层含义分析：

1. 本地已有 AAAI 稿件：`Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval`，简称 NIRNL；
2. 截至 2026 年 6 月公开的最新直接相关 AAAI 工作，重点是 AAAI 2026 的 SCBCH，并补充 AAAI 2025 的 ACHFCA 和 AAAI 2026 的 DPSIH。

当前判断如下：

- SSMH 与 NIRNL 的研究问题不同，能够形成连续研究路线，而不是简单替换 NIRNL。
- NIRNL 解决“噪声标签是否可靠”，SSMH 解决“多标签语义是否足够细、能否在汉明空间保真”。
- SSMH 在当前 NUS-WIDE-TC21 实验中明显优于 NIRNL，但现有结果仍是机制验证，不能直接作为正式跨模态哈希论文的 SOTA 对比。
- AAAI 2026 的 SCBCH 与当前方向高度重合，是必须正面对比的第一强基线。
- 当前 SSMH 若只强调 Jaccard、软相似度和噪声多标签，创新性会被 SCBCH、DSFH 等工作覆盖。
- SSMH 最值得保留和加强的差异是：**标签结构图 + 语义自适应边际 + 多标签组合原型 + 分级语义汉明保真评价**。

---

## 1. 当前 SSMH 与 NIRNL AAAI 稿件的研究定位对比

| 对比维度 | NIRNL | 当前 SSMH | 关系与判断 |
|---|---|---|---|
| 核心问题 | 标签错误、噪声监督导致模型误学习 | 多标签部分匹配和语义重叠被硬相似度抹平 | 两者问题互补 |
| 主要假设 | 标签可能不可靠，但类别语义关系主要按硬标签处理 | 标签可形成连续、分级的语义相关程度 | SSMH 应避免默认依赖干净标签 |
| 监督对象 | pure、hard、noisy 三类训练实例 | 样本对软相似度、类别图、组合原型 | 从实例可靠性转向语义结构 |
| 核心模块一 | Cross-modal Margin Preserving，固定边际增强正负对判别 | Soft Margin，根据语义相似度调整样本对约束 | SSMH 是对 CMP 的语义化扩展 |
| 核心模块二 | 邻域一致性划分 pure/hard/noisy 子集 | Jaccard、overlap、cosine 与标签图融合 | NIRNL 关注可靠性，SSMH 关注相关程度 |
| 类别结构 | Wasserstein barycenter 和 KNN 软标签 | 类别 prototype 与多标签组合 prototype | SSMH 更适合复合语义 |
| 数据利用策略 | 对三个子集采用不同优化策略，尽量不丢弃数据 | 对不同语义强度的样本对采用连续约束 | 都强调充分利用数据，但作用层级不同 |
| 噪声鲁棒性 | 主任务，实验噪声率为 0.2/0.4/0.6/0.8 | 当前只是附加目标，尚未形成可靠标签修正闭环 | 不能声称已经超过 NIRNL 的噪声能力 |
| 多标签建模 | 不是主要创新点 | 主要研究对象 | SSMH 应以多标签数据集为主 |
| 检索空间 | 当前 NIRNL 稿件主要是连续公共表示空间 | 目标是二值汉明空间 | 当前 SSMH 代码尚未完整落实这一差别 |
| 评价指标 | MAP、PR 曲线 | MAP、nDCG、weighted MAP、语义-Hamming 相关性 | SSMH 的评价更能验证分级语义 |
| 推荐论文叙事 | 可靠监督提纯 | 细粒度语义度量与哈希保真 | 可组成一条连续博士研究主线 |

### 1.1 两项工作的连续研究路线

```text
NIRNL：识别监督是否可靠
          ↓
得到更可信的实例或标签置信度
          ↓
SSMH：刻画可信标签之间的细粒度关系
          ↓
将分级语义保留到紧凑二值码
```

因此，第二项工作不宜表述为“在 NIRNL 上增加几个损失”，而应表述为：

> NIRNL 面向噪声监督下的实例可靠性学习；SSMH 面向多标签复杂语义下的分级关系建模与汉明空间保真。前者回答监督能否相信，后者回答可信监督是否足够精细。

---

## 2. 当前实验中 SSMH 与 NIRNL 的数值对比

以下结果来自当前仓库、当前特征和当前评价代码，不等同于原 AAAI 稿件表格中的正式协议。

| 数据集 | NIRNL Avg MAP | SSMH Avg MAP | 绝对变化 | 相对变化 | 当前解释 |
|---|---:|---:|---:|---:|---|
| `wiki` | 0.505195 | 0.505342 | +0.000147 | +0.03% | 基本持平，单标签数据无法体现 SSMH |
| `xmedia` | 0.917662 | 0.919111 | +0.001449 | +0.16% | 小幅提升，证据较弱 |
| `INRIA-Websearch` | 0.526144 | 0.471602 | -0.054542 | -10.37% | 100 类单标签且正样本稀疏，标签图和软 pair 可能负贡献 |
| `NUS-WIDE-TC21` | 0.542856 ± 0.001836 | 0.586678 ± 0.002577 | +0.043822 ± 0.003436 | +8.07% | 三个 seed 稳定提升，是当前最有力结果 |

### 2.1 可以得出的结论

- SSMH 的优势目前只在真正多标签的 NUS-WIDE-TC21 上稳定出现。
- `wiki`、`xmedia`、`INRIA-Websearch` 主要是单标签数据，不能作为 SSMH 主要创新的核心证据。
- INRIA 的下降说明当前标签图、soft pair 和 prototype 不能无条件应用于单标签场景。
- 当前结果支持“SSMH 是多标签语义方法”，但尚不足以支持“SSMH 是先进的跨模态哈希方法”。

### 2.2 不能直接写入论文的结论

- 不能把当前结果直接与 SCBCH、ACHFCA、PromptHash 的论文数值横向比较。
- 不能用当前结果声称达到跨模态哈希 SOTA。
- 不能用 `ssmh_use_clean_labels=True` 的结果证明噪声标签鲁棒性。
- 不能把 512 维连续 cosine 检索结果称为 512-bit 哈希结果。

---

## 3. 当前 SSMH 与最新 AAAI 直接相关工作的对比

### 3.1 与 SCBCH（AAAI 2026）的对比

SCBCH 全称为 **Semantic-Consistent Bidirectional Contrastive Hashing for Noisy Multi-Label Cross-Modal Retrieval**。这是当前最接近 SSMH 研究设定的方法。

| 对比维度 | SCBCH | 当前 SSMH | 重合风险与应对 |
|---|---|---|---|
| 任务 | 噪声多标签跨模态哈希 | 多标签复杂语义跨模态哈希，并计划考虑噪声 | 高度重合 |
| 软语义来源 | 标签重叠比例构造软相似度 | Jaccard、overlap、cosine 加权，可选标签图传播 | SSMH 必须证明结构图优于简单重叠 |
| 标签噪声处理 | semantic consistency 进行标签细化 | 当前默认可直接使用 clean labels，尚无可靠修正机制 | SCBCH 当前明显更完整 |
| 对比学习 | 软语义监督的 bidirectional contrastive loss | adaptive soft-margin rank loss + pair preserving loss | 可形成损失设计差异 |
| 全局类别结构 | 主要围绕语义一致性与双向对比 | 多标签组合 prototype | 这是 SSMH 的重要差异点 |
| 汉明空间 | 正式多码长二值哈希评估 | 有量化、平衡、去相关损失，但主 MAP 仍用连续 cosine | 必须补齐二值检索协议 |
| 数据集 | MIRFlickr-25K、NUS-WIDE、MS-COCO、IAPR-TC12 | 当前只有自建 NUS-WIDE-TC21 和单标签旧数据 | 必须对齐公开协议 |
| 噪声率 | 0.2、0.4、0.6 | 当前主要完成 0.2 | 需要补完整曲线 |
| 码长 | 16、32、64、128 bits | 默认 `output_dim=512` | 必须整改 |

### 与 SCBCH 拉开差异的必要条件

SSMH 至少需要同时满足下面三点中的两点，才有较清晰的新颖性：

1. **结构化软语义**：不只使用标签重叠比例，而是显式利用标签共现图、标签嵌入或层级关系；
2. **多标签组合原型**：用样本特定的组合 prototype 建模全局复合语义，区别于仅做 pairwise 对比；
3. **分级汉明保真**：直接优化和评估软语义排序在二值码中的保持程度，而不仅报告普通 MAP。

如果最终方法仍主要是“标签 Jaccard + 对比损失 + 噪声标签”，则与 SCBCH 的差异不足。

### 3.2 与 ACHFCA（AAAI 2025）的对比

ACHFCA 使用形式概念分析，从对象、属性和概念层级中重构细粒度跨模态相似关系。

| ACHFCA 的优势 | SSMH 可形成的差异 |
|---|---|
| 显式挖掘概念层级和对象-属性结构 | 用可微标签图和组合原型实现更轻量的端到端结构建模 |
| 直接针对粗粒度二值相似矩阵的缺陷 | 进一步让结构化相似度控制动态边际和汉明排序 |
| 在四个跨模态哈希数据集上采用标准多码长协议 | SSMH 必须对齐其数据和评估协议 |

ACHFCA 必须进入主表，因为它会挑战 SSMH 的“结构化细粒度语义”创新点。

### 3.3 与 DPSIH（AAAI 2026）的对比

DPSIH 关注 polysemic semantic instances，为多目标图像和含歧义文本生成多个语义嵌入。

| DPSIH | SSMH |
|---|---|
| 解决一个样本内部存在多个潜在语义的问题 | 解决多个样本之间多标签相关程度不同的问题 |
| 一样本多嵌入 | 一样本一个组合原型和一个哈希码 |
| 强调实例内部语义解耦 | 强调样本间分级关系与汉明保真 |

DPSIH 不是第一优先级基线，但适合进入扩展实验或相关工作，用于说明 SSMH 并不显式建模实例内部多义性。

### 3.4 与 DSFH（IJCAI 2024）的对比

DSFH 融合粗粒度 pairwise 语义和基于多标签分布的细粒度 cluster-level 语义。

重合点：

- 都反对仅使用 0/1 样本对关系；
- 都构造细粒度、多层语义监督；
- 都强调哈希码判别性。

SSMH 必须证明：

- 标签图或多源结构监督优于 DSFH 的细粒度语义构造；
- adaptive margin 和 composite prototype 各自带来独立增益；
- 优势不仅体现在普通 MAP，还体现在 nDCG、weighted MAP 和 Hamming 语义相关性。

### 3.5 与 PromptHash（CVPR 2025）的对比

PromptHash 利用 affinity prompts、跨模态门控融合和层级对比学习提升自适应哈希检索。

PromptHash 的特征提取和语义先验更强，适合作为现代强基线。SSMH 的差异不应放在大模型或 prompt，而应强调：

- 不依赖额外提示生成；
- 结构化标签监督更透明；
- 组合原型和动态边际更适合解释多标签相关程度；
- 在相同预训练特征下仍能获得提升。

---

## 4. 当前代码与正式哈希论文协议之间的差距

这是开展 SOTA 对比之前必须修正的部分。

| 当前问题 | 代码现状 | 对论文比较的影响 | 必须修改 |
|---|---|---|---|
| 不是正式二值 MAP | `fx_calc_map_multilabel` 使用连续特征的 cosine distance | 不能与基于 Hamming distance 的论文表格公平比较 | 增加 sign/binary code 的 Hamming MAP、P@K、PR |
| 码长不规范 | 默认 `output_dim=512` | 与 16/32/64/128-bit 协议不一致 | 主实验改为 16/32/64/128 |
| 干净标签泄漏 | `ssmh_use_clean_labels=True` 时，soft similarity、soft margin、标签图和 prototype 都使用原始干净标签 | 噪声实验不成立，现实训练中不可获得干净标签 | 正式噪声实验必须设为 False，或先设计标签修正模块 |
| 数据划分不一致 | 当前 NUS-WIDE-TC21 为自建 10500/2100/2100 | 不能直接比较公开论文数值 | 增加与 SCBCH/ACHFCA 一致的标准 split |
| 特征不一致 | 当前 NUS-WIDE 使用 634 维 low-level image + 1000 维 tags | 与 VGG19/BERT 或其他论文特征不一致 | 至少建立一个统一 VGG/BERT 协议 |
| 缺少现代多标签数据 | 当前主实验仍混入三个单标签数据集 | 不能充分支持复杂多标签语义主张 | 增加 MIRFlickr-25K、MS-COCO，保留 NUS-WIDE |
| 未完成关键消融 | full 模型已经跑通，但模块独立贡献未验证 | 难以证明创新来自哪个模块 | 补 SoftSim、Graph、Margin、Prototype、HashReg 消融 |

### 当前结果的正确命名

在完成上述修改前，建议把已有结果称为：

> continuous-representation mechanism validation under the NIRNL feature protocol

中文可写为：

> 基于 NIRNL 特征协议的连续表示机制验证结果

不要暂时称为“标准跨模态哈希主实验结果”。

---

## 5. 当前工作需要对比的方法

### 5.1 第一优先级：必须进入外部方法主表

| 方法 | 年份/会议 | 类型 | 为什么必须比较 |
|---|---|---|---|
| DCMH | CVPR 2017 | 经典监督跨模态哈希 | 最基本的共同基线，验证新模块是否优于标准 pairwise hashing |
| DJSRH | ICCV 2019 | 经典无监督联合语义哈希 | 提供无监督参照 |
| UCCH | TPAMI 2023 | 无监督对比跨模态哈希 | NIRNL 和 SCBCH 都使用的现代无监督强基线 |
| CPAH | TIP 2020 | 监督对抗哈希 | 多篇新工作共同使用，便于横向对齐 |
| CMMQ | ACM MM 2022 | 监督跨模态量化 | SCBCH 官方协议采用的方法 |
| MIAN | TOMM 2023 | 监督跨模态哈希 | SCBCH 官方协议采用的方法 |
| LtCMH | TCSVT 2023 | 监督跨模态哈希 | SCBCH 官方协议采用的方法 |
| DSFH | IJCAI 2024 | 多标签细粒度语义哈希 | 与 SSMH 的软语义创新最直接相关 |
| ACHFCA | AAAI 2025 | 结构化细粒度相似度哈希 | 与标签结构和细粒度关系高度相关 |
| PromptHash | CVPR 2025 | 提示驱动现代跨模态哈希 | 代表强预训练与层级语义路线 |
| SCBCH | AAAI 2026 | 噪声多标签跨模态哈希 | 当前最直接、最强、最不能缺少的竞争方法 |

上述 11 个方法构成较完整的外部方法主表。如果复现成本过高，最低限度保留：

```text
DCMH, UCCH, CPAH, DSFH, ACHFCA, PromptHash, SCBCH
```

### 5.2 必须单独保留的内部与研究衔接基线

| 方法 | 使用方式 | 原因 |
|---|---|---|
| NIRNL | 在相同特征、相同 split 下与 SSMH 比较 | 证明第二项工作相对第一项工作的增益与边界 |
| NIRNL-Hash | 在 NIRNL 后增加相同二值层和量化约束 | 若要进入正式哈希主表，需要先把 NIRNL 改成公平的哈希版本 |
| SSMH-Hard | 仅使用二值相似度 | 证明软语义监督的必要性 |
| SSMH-Jaccard | 只使用标签 Jaccard | 证明结构图和多源语义不是装饰模块 |

原始 NIRNL 是连续跨模态检索方法时，建议放在“与前期工作的衔接实验”中，不要和正式二值哈希方法混在同一张 SOTA 主表。只有 NIRNL-Hash 才适合进入相同码长的主表。

### 5.3 第二优先级：建议进入扩展表或噪声实验

| 方法 | 类型 | 使用位置 |
|---|---|---|
| RONO | 噪声鲁棒跨模态检索 | 与 NIRNL 协议对齐的噪声实验 |
| NRCH | 噪声鲁棒跨模态哈希 | 噪声率曲线和鲁棒性表 |
| DHRL | 噪声鲁棒跨模态检索 | 与 SCBCH/NIRNL 的高噪声结果对齐 |
| RSHNL | AAAI 2025 噪声鲁棒检索 | 与 NIRNL、SCBCH 共同比较 |
| NACD | NeurIPS 2025 冗余标注消歧哈希 | 标签冗余或候选标签噪声实验 |
| DPSIH | AAAI 2026 多义实例哈希 | MS-COCO 或多目标语义扩展实验 |
| WCMH | 标签增强和加权汉明距离 | graded relevance 和距离度量相关实验 |
| DSPH | 多标签 proxy hashing | 对比组合 prototype 的必要性 |
| DSCGH/DCGH | semantic center/class-guided hashing | 对比单中心与组合原型 |

### 5.4 不建议作为 SSMH 主表核心的方法

NIRNL 稿件中的 ALGCN、GNN4CMR、DRCL、DHRL 等方法可以保留在与第一项工作的衔接实验中，但它们多数不是直接针对“多标签细粒度哈希”的方法。

因此建议：

- 不要把 NIRNL 原来的 10 个基线原样搬到 SSMH 主表；
- 主表优先采用跨模态哈希和多标签语义方法；
- 噪声附表再放 RONO、NRCH、RSHNL、NACD、NIRNL；
- 单标签 `wiki/xmedia/INRIA` 结果放附录或“跨场景泛化”小节。

---

## 6. 推荐的正式实验协议

### 6.1 数据集

| 优先级 | 数据集 | 作用 |
|---:|---|---|
| 1 | MIRFlickr-25K | 快速开发、消融和参数敏感性 |
| 2 | NUS-WIDE | 大规模多标签主实验 |
| 3 | MS-COCO | 复杂对象组合和多描述语义验证 |
| 4 | IAPR-TC12 | 与 ACHFCA、SCBCH 对齐的补充数据集 |

当前自建 `NUS-WIDE-TC21 low-level+tags` 可以保留，但应标记为额外协议，而不是唯一 NUS-WIDE 结果。

### 6.2 码长

```text
16, 32, 64, 128 bits
```

若算力紧张，第一轮先跑：

```text
32, 64 bits
```

趋势成立后再补齐 16 和 128 bits。

### 6.3 指标

主表：

- MAP@all 或论文统一的 MAP@K；
- I2T 和 T2I；
- P@K；
- PR curve。

复杂语义表：

- nDCG@K；
- weighted MAP；
- Hamming distance 与 Jaccard relevance 的 Spearman/Kendall 相关性；
- 不同标签重叠等级下的平均 Hamming distance。

效率表：

- 训练时间；
- 查询时间；
- 参数量；
- 二值码存储开销。

### 6.4 噪声协议

与 SCBCH 对齐：

```text
noise ratio = 0.2, 0.4, 0.6
```

建议额外保留：

```text
noise ratio = 0.0
```

这样可以区分：

- 干净多标签语义建模能力；
- 轻度噪声鲁棒性；
- 高噪声下的退化速度。

---

## 7. 推荐消融实验

| 变体 | 修改 | 回答的问题 |
|---|---|---|
| SSMH-Hard | 只使用二值相似矩阵 | 软语义是否必要 |
| Jaccard-only | 只用 Jaccard | 多源结构语义是否优于简单标签重叠 |
| w/o Graph | 关闭标签图 | 标签共现结构是否有效 |
| w/o SoftMargin | 固定 margin | 语义自适应边际是否有效 |
| w/o SoftPair | 去掉 pair preserving loss | 直接拟合软相似度是否有效 |
| w/o Prototype | 去掉组合原型 | 全局复合类别结构是否有效 |
| Single Prototype | 多标签样本只分配一个中心 | 组合原型是否优于单中心 |
| w/o Quant | 去掉量化约束 | 连续到二值的误差控制是否有效 |
| w/o Balance/Decor | 去掉位平衡和去相关 | 哈希位质量是否有效 |
| Noisy-label semantics | 用 noisy labels 构图和原型 | 测试真实噪声场景 |
| Refined-label semantics | 用 NIRNL/新校正模块产生的标签构图 | 验证两项工作的真正结合 |

最关键的最小消融集合：

```text
Full SSMH
SSMH-Hard
Jaccard-only
w/o SoftMargin
w/o Prototype
w/o Hash Regularization
```

---

## 8. 下一步执行顺序

### 阶段 A：先把协议改正确

1. 增加真正的 binary Hamming MAP、P@K 和 PR 评价；
2. 将输出维度切换为 16/32/64/128 bits；
3. 将正式噪声实验的 `ssmh_use_clean_labels` 设为 False；
4. 保留 clean-label 结果，但明确标记为 oracle 上界；
5. 固定标准数据 split、特征和随机种子。

### 阶段 B：先跑最小可发表比较

优先使用 MIRFlickr-25K 和 NUS-WIDE：

```text
DCMH
UCCH
DSFH
ACHFCA
SCBCH
NIRNL
SSMH-Hard
Full SSMH
```

码长先跑 32/64 bits，噪声率先跑 0.0/0.2/0.4。

### 阶段 C：补全正式论文

1. 增加 PromptHash、NACD、DPSIH；
2. 增加 MS-COCO 和 IAPR-TC12；
3. 补 16/128 bits；
4. 完成所有消融、参数敏感性和效率实验；
5. 报告三个 seed 的 `mean ± std`。

---

## 9. 最终创新定位建议

不建议使用下面这种表述：

> 我们通过标签重叠构造软相似度，并设计对比损失进行跨模态哈希。

该表述与 DSFH、SCBCH 等工作过于接近。

建议收敛为：

> 本研究提出结构感知软边际跨模态哈希框架。不同于仅依据标签重叠构造软相似度的方法，我们联合标签共现结构与多标签组合原型形成样本级和类别级双层语义监督，并以语义相关程度自适应调节跨模态排序边际。进一步地，模型显式约束分级语义关系在二值汉明空间中的排序一致性，从而同时改善多标签部分匹配、全局复合类别结构和最终哈希码的语义保真。

### 一句话区分主要竞争方法

- 相比 NIRNL：从“标签是否可靠”推进到“可靠语义是否足够细”；
- 相比 DSFH：从多层语义融合推进到结构图、动态边际和组合原型联合建模；
- 相比 ACHFCA：用端到端可微结构监督替代较重的形式概念构造，并加入汉明排序保真；
- 相比 SCBCH：从标签重叠与双向对比推进到类别结构、组合原型和 graded Hamming preservation；
- 相比 PromptHash：不依赖 prompt 生成，强调显式可解释的多标签结构监督。

---

## 10. 参考资料

1. 本地 NIRNL 稿件：`Neighbor_aware_Instance_Refining_with_Noisy_Labels_for_Cross_Modal_Retrieval (4).pdf`
2. [DSFH: Dual Semantic Fusion Hashing for Multi-Label Cross-Modal Retrieval, IJCAI 2024](https://www.ijcai.org/proceedings/2024/505)
3. [ACHFCA: Asymmetric Cross-Modal Hashing Based on Formal Concept Analysis, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32129)
4. [PromptHash: Affinity-Prompted Collaborative Cross-Modal Learning for Adaptive Hashing Retrieval, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zou_PromptHashAffinity-Prompted_Collaborative_Cross-Modal_Learning_for_Adaptive_Hashing_Retrieval_CVPR_2025_paper.html)
5. [NACD: Neighbor-Aware Contrastive Disambiguation for Cross-Modal Hashing with Redundant Annotations, NeurIPS 2025](https://openreview.net/forum?id=Bi1udlTjMb)
6. [SCBCH: Semantic-Consistent Bidirectional Contrastive Hashing for Noisy Multi-Label Cross-Modal Retrieval, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39667)
7. [DPSIH: Deep Polysemic Semantic Instance Hashing for Cross-Modal Retrieval, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/42459)
