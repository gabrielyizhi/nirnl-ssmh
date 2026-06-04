# SSMH 实验计划与结果追踪

更新时间：2026-05-27

本文档用于追踪 `nirnl-ssmh` 后续所有实验。每次实验跑完后，需要在“结果记录”中追加一行，并在“阶段状态”中更新完成情况。

## 当前结论

- 服务器当前可用数据集：`wiki`、`xmedia`、`INRIA-Websearch`、`nuswide`。
- `nuswide` 已经根据官方 NUS-WIDE Train/Test split、1k tags 文本特征和 5 组 normalized low-level 图像特征构建为代码可读的 h5py 文件。
- 当前 NUS-WIDE 文件是可运行版本，不是原论文历史使用的 deep/doc2vec 预处理文件；因此首次结果应标注为 `NUS-WIDE-TC21 low-level+tags`。
- 主结果矩阵已经完成 `wiki/xmedia/INRIA-Websearch` 的 NIRNL/SSMH 默认对比；NUS-WIDE-TC21 上也已完成 NIRNL/SSMH 默认对比。
- NUS-WIDE-TC21 上 SSMH 明显领先 NIRNL：Avg MAP 从 `0.540769` 提升到 `0.586722`，Hamming semantic Spearman 从 `0.283285` 提升到 `0.424776`。这说明真正多标签数据更能体现 SSMH 的软语义建模优势。

## 服务器数据集状态

数据根目录：

```bash
/home/liuyizhi/NIRNL-AAAI26/Clean_idx
```

| 数据集参数名 | 代码期望文件 | 服务器状态 | 备注 |
|---|---|---|---|
| `wiki` | `wiki.mat` | 可用 | 已完成 NIRNL/SSMH 默认主结果 |
| `xmedia` | `XMediaFeatures.mat` | 可用 | 已完成 NIRNL/SSMH 默认主结果 |
| `INRIA-Websearch` | `INRIA-Websearch.mat` | 可用 | 已完成 NIRNL/SSMH 默认主结果，SSMH 落后需定位 |
| `nuswide` | `nus_wide_deep_doc2vec-corr-ae.h5py` | 可用 | 2026-05-30 新构建的 TC21 low-level+tags 版本，已完成 NIRNL/SSMH 默认对比 |
| `xmedianet` | `xmedianet_deep_doc2vec_data.h5py` + `XMediaNet5View_Doc2Vec.mat` | 缺失 | 可作为后续扩展数据集 |

只读搜索结果：

- `/home/liuyizhi/NIRNL-AAAI26/Clean_idx/nus_wide_deep_doc2vec-corr-ae.h5py` 已生成。
- 未找到 `/home/liuyizhi/**/xmedianet_deep_doc2vec_data.h5py`。
- 未找到 `/home/liuyizhi/**/XMediaNet5View_Doc2Vec.mat`。
- 找到了旧实验输出，例如 `/home/liuyizhi/NIRNL-AAAI26/Avg_MAP_data/nuswide_Avg_MAP_0.2_0_.mat` 和 `/home/liuyizhi/NIRNL-AAAI26/logging/nuswide.log`，说明历史上可能在别的路径或别的机器上跑过 NUS-WIDE。

## 阶段状态

| 阶段 | 目标 | 状态 | 下一步 |
|---|---|---|---|
| 环境验证 | 服务器连接、GPU、依赖、代码入口跑通 | 完成 | 无 |
| Baseline 验证 | WIKI + NIRNL + 默认 AAAI 超参 | 完成 | 作为第一条 baseline |
| 主方法验证 | WIKI + SSMH + 同等超参 | 完成 | 无 |
| 主结果表 | 3 个现有数据集上比较 NIRNL 和 SSMH | 完成 | 已扩展 NUS-WIDE-TC21 默认对比 |
| 噪声鲁棒性 | 多噪声比例比较 NIRNL 和 SSMH | 待做 | 先 WIKI，后扩展 |
| 消融实验 | 验证 SSMH 各模块贡献 | 待做 | 先 WIKI r=0.2/r=0.4 |
| 多 seed | 统计稳定性 `mean ± std` | 待做 | 主趋势稳定后补 |
| NUS-WIDE | 获取或构建 NUS-WIDE 预处理特征 | 完成一版并跑通 | 下一步做多 seed/噪声比例，或补更强深度特征 |

## 第一阶段：必须先跑的主结果

目标：先用服务器已有数据集建立 `NIRNL vs SSMH` 的主对比。

默认训练参数：

```text
MAX_EPOCH = 100
batch_size = 256
lr = 1e-4
output_dim = 512
noisy_ratio = 0.2
noise_mode = sym
seed = 1
```

### 已完成

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --method nirnl \
  --dataset wiki \
  --data_root /home/liuyizhi/NIRNL-AAAI26/Clean_idx \
  --noise_root /home/liuyizhi/nirnl-ssmh/noisy \
  --logging aaaidefault_wiki_nirnl
```

### 下一条立即运行

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --method ssmh \
  --dataset wiki \
  --data_root /home/liuyizhi/NIRNL-AAAI26/Clean_idx \
  --noise_root /home/liuyizhi/nirnl-ssmh/noisy \
  --logging aaaidefault_wiki_ssmh
```

### 主结果矩阵

| 优先级 | 数据集 | 方法 | 噪声比例 | seed | 状态 |
|---:|---|---|---:|---:|---|
| 1 | `wiki` | `nirnl` | 0.2 | 1 | 完成 |
| 2 | `wiki` | `ssmh` | 0.2 | 1 | 完成 |
| 3 | `xmedia` | `nirnl` | 0.2 | 1 | 完成 |
| 4 | `xmedia` | `ssmh` | 0.2 | 1 | 完成 |
| 5 | `INRIA-Websearch` | `nirnl` | 0.2 | 1 | 完成 |
| 6 | `INRIA-Websearch` | `ssmh` | 0.2 | 1 | 完成 |

## 第二阶段：噪声鲁棒性实验

目标：证明 SSMH 在标签噪声增加时比 NIRNL 更稳。

建议噪声比例：

```text
0.0, 0.1, 0.2, 0.4, 0.6
```

优先顺序：

| 优先级 | 数据集 | 方法 | 噪声比例 |
|---:|---|---|---|
| 1 | `wiki` | `nirnl`, `ssmh` | 0.0, 0.1, 0.2, 0.4, 0.6 |
| 2 | `xmedia` | `nirnl`, `ssmh` | 0.0, 0.1, 0.2, 0.4, 0.6 |
| 3 | `INRIA-Websearch` | `nirnl`, `ssmh` | 0.0, 0.1, 0.2, 0.4, 0.6 |

命令模板：

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --method METHOD \
  --dataset DATASET \
  --noisy_ratio RATIO \
  --data_root /home/liuyizhi/NIRNL-AAAI26/Clean_idx \
  --noise_root /home/liuyizhi/nirnl-ssmh/noisy \
  --logging DATASET_METHOD_rRATIO_s1
```

## 第三阶段：SSMH 消融实验

目标：验证每个创新模块是否真的有效。

优先在 `wiki` 上跑，噪声比例先用 `0.2` 和 `0.4`。

| 实验名 | 目的 | 参数 |
|---|---|---|
| Full SSMH | 完整方法 | 默认 SSMH |
| w/o semantic graph | 去掉标签图传播 | `--use_label_graph False` |
| w/o prototype | 去掉语义原型约束 | `--prototype_weight 0` |
| w/o soft pair | 去掉软语义成对监督 | `--soft_pair_weight 0` |
| w/o quantization | 去掉量化约束 | `--quant_weight 0` |
| label-only similarity | 只用标签相似度 | `--semantic_sim label` |
| jaccard-only similarity | 只用 Jaccard | `--semantic_sim jaccard` |

最小消融集合：

```text
Full SSMH
w/o semantic graph
w/o prototype
w/o soft pair
```

## 第四阶段：多 seed 稳定性

目标：正式论文表格建议报告 `mean ± std`。

建议 seed：

```text
1, 2, 3
```

优先补：

| 数据集 | 方法 | 噪声比例 | seed |
|---|---|---:|---|
| `wiki` | `nirnl`, `ssmh` | 0.2 | 1, 2, 3 |
| `wiki` | `nirnl`, `ssmh` | 0.4 | 1, 2, 3 |
| `xmedia` | `nirnl`, `ssmh` | 0.2 | 1, 2, 3 |
| `xmedia` | `nirnl`, `ssmh` | 0.4 | 1, 2, 3 |

## NUS-WIDE 数据集计划

### 当前状态

代码的 `nuswide` loader 不是读取原始图片，而是读取已经预处理好的特征文件：

```text
nus_wide_deep_doc2vec-corr-ae.h5py
```

这个文件需要包含以下键：

```text
train_imgs_deep
train_texts
train_imgs_labels
valid_imgs_deep
valid_texts
valid_imgs_labels
test_imgs_deep
test_texts
test_imgs_labels
```

2026-05-30 已经用本地下载的 5 个 NUS-WIDE 压缩包构建出一版可运行 h5py：

```text
/home/liuyizhi/NIRNL-AAAI26/Clean_idx/nus_wide_deep_doc2vec-corr-ae.h5py
```

构建方式：

- 原始压缩包存放在 `/home/liuyizhi/datasets/NUS-WIDE_raw`。
- 解压后的工作目录是 `/home/liuyizhi/datasets/NUS-WIDE_work`。
- 转换脚本是 `scripts/build_nuswide_h5py.py`。
- 文本特征使用官方 `Train_Tags1k.dat` / `Test_Tags1k.dat`，维度 1000。
- 图像特征使用 `CH + CM55 + CORR + EDH + WT` 五组 normalized low-level features 拼接，维度 634。
- 标签使用官方 TrainTestLabels 中样本频率最高的 21 类，保留至少有一个所选标签的样本。
- 当前 split：train 10500、valid 2100、test 2100。
- 已验证 h5py key/shape，并已通过 `load_data.get_loader('nuswide')`。

当前 21 个标签：

```text
sky, clouds, person, water, animal, grass, buildings, window, plants, ocean,
road, flowers, sunset, reflection, rocks, vehicle, snow, tree, beach, mountain, boats
```

### 建议

短期可以先用这版 `NUS-WIDE-TC21 low-level+tags` 跑通 NIRNL/SSMH 主流程，用来验证多标签语义模块是否比单标签数据更适配。论文正式表述时要注明这不是原论文历史的 deep/doc2vec 特征版本；如果后续需要更强结果，再补 CLIP/ResNet 图像特征和更强文本语义特征。

## 结果记录

每次实验跑完后，在这里新增一行。

| 日期 | 数据集 | 方法 | 噪声比例 | seed | 额外参数 | I2T MAP | T2I MAP | Avg MAP | I2T nDCG@100 | T2I nDCG@100 | Hamming Spearman | 日志名 | 状态 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-05-27 | `wiki` | `nirnl` | 0.2 | 1 | AAAI default | 0.523821 | 0.486570 | 0.505195 | 0.655180 | 0.670087 | 0.313426 | `aaaidefault_wiki_nirnl` | 完成 |
| 2026-05-28 | `wiki` | `ssmh` | 0.2 | 1 | AAAI default + SSMH default | 0.530816 | 0.479868 | 0.505342 | 0.649357 | 0.669582 | 0.321967 | `aaaidefault_wiki_ssmh` | 完成 |
| 2026-05-28 | `xmedia` | `nirnl` | 0.2 | 1 | AAAI default | 0.918511 | 0.916813 | 0.917662 | 0.941751 | 0.958052 | 0.357104 | `aaaidefault_xmedia_nirnl` | 完成 |
| 2026-05-28 | `xmedia` | `ssmh` | 0.2 | 1 | AAAI default + SSMH default | 0.920312 | 0.917910 | 0.919111 | 0.945662 | 0.958071 | 0.358795 | `aaaidefault_xmedia_ssmh` | 完成 |
| 2026-05-28 | `INRIA-Websearch` | `nirnl` | 0.2 | 1 | AAAI default | 0.521811 | 0.530476 | 0.526144 | 0.590407 | 0.630620 | 0.110692 | `aaaidefault_inria_nirnl` | 完成 |
| 2026-05-28 | `INRIA-Websearch` | `ssmh` | 0.2 | 1 | AAAI default + SSMH default | 0.468335 | 0.474869 | 0.471602 | 0.557453 | 0.588609 | 0.122077 | `aaaidefault_inria_ssmh` | 完成 |
| 2026-06-04 | `nuswide` | `nirnl` | 0.2 | 1 | NUS-WIDE-TC21 low-level+tags, AAAI default | 0.545773 | 0.535765 | 0.540769 | 0.445470 | 0.468397 | 0.283285 | `aaaidefault_nuswide_tc21_nirnl` | 完成 |
| 2026-06-04 | `nuswide` | `ssmh` | 0.2 | 1 | NUS-WIDE-TC21 low-level+tags, AAAI default + SSMH default | 0.590982 | 0.582462 | 0.586722 | 0.439560 | 0.461664 | 0.424776 | `aaaidefault_nuswide_tc21_ssmh` | 完成 |

## NUS-WIDE-TC21 默认结果分析

### 现象

在新构建的 `NUS-WIDE-TC21 low-level+tags` 上，SSMH 明显优于 NIRNL：

| 数据集 | 方法 | Avg MAP | I2T MAP | T2I MAP | nDCG 平均 | Hamming Spearman |
|---|---|---:|---:|---:|---:|---:|
| `nuswide` | `nirnl` | 0.540769 | 0.545773 | 0.535765 | 0.456935 | 0.283285 |
| `nuswide` | `ssmh` | 0.586722 | 0.590982 | 0.582462 | 0.450612 | 0.424776 |

关键结论：

- SSMH 的 Avg MAP 相比 NIRNL 提升 `+0.045953`，I2T/T2I 两个方向都明显提升。
- Hamming semantic Spearman 提升 `+0.141491`，说明 SSMH 的哈希空间语义排序保持能力显著更强。
- nDCG@100 略低于 NIRNL，说明前 100 个近邻的局部排序仍有优化空间；当前 SSMH 更突出的是整体相关样本召回和哈希语义结构。
- 这组结果支持我们的核心判断：`wiki/xmedia/INRIA` 多为单标签或弱多标签场景，难以充分体现 SSMH；真正多标签 NUS-WIDE-TC21 上，软语义、标签图和组合原型开始发挥作用。

### 下一步

建议优先补 NUS-WIDE-TC21 的稳定性和鲁棒性：

| 优先级 | 实验 | 目的 |
|---:|---|---|
| 1 | `nuswide` seed 2/3，NIRNL vs SSMH，噪声 0.2 | 验证 `+0.046 Avg MAP` 是否稳定 |
| 2 | `nuswide` 噪声比例 0.0/0.1/0.4/0.6 | 验证多标签噪声鲁棒性 |
| 3 | `nuswide` SSMH 消融 | 验证标签图、soft pair、prototype 在多标签数据上的贡献 |
| 4 | 更强图像/文本特征版本 | 用 ResNet/CLIP 图像特征和更强文本特征提升正式论文结果 |

## INRIA-Websearch 落后分析

### 现象

在 `INRIA-Websearch` 上，默认 SSMH 明显落后于 NIRNL：

| 数据集 | 方法 | Avg MAP | I2T MAP | T2I MAP | nDCG 平均 | Hamming Spearman |
|---|---|---:|---:|---:|---:|---:|
| `INRIA-Websearch` | `nirnl` | 0.526144 | 0.521811 | 0.530476 | 0.610514 | 0.110692 |
| `INRIA-Websearch` | `ssmh` | 0.471602 | 0.468335 | 0.474869 | 0.573031 | 0.122077 |

SSMH 的 Hamming Spearman 略高，说明它对语义距离排序有一点改善，但 mAP 和 nDCG 明显下降。这说明当前默认 SSMH 更像是在优化连续语义结构，而没有充分优化检索排序中的同类正样本召回。

### 数据层面证据

对服务器已有三个数据集的训练标签做只读统计后发现：

| 数据集 | 训练样本 | 类别数 | 平均标签数 | 随机样本对零重叠比例 | Jaccard 均值 |
|---|---:|---:|---:|---:|---:|
| `wiki` | 2173 | 10 | 1.0 | 0.8923 | 0.1077 |
| `xmedia` | 4000 | 21 | 1.0 | 0.9498 | 0.0502 |
| `INRIA-Websearch` | 9000 | 100 | 1.0 | 0.9897 | 0.0103 |

关键结论：

- 当前三个数据集都不是严格意义上的多标签数据，都是单标签或被处理成单标签。
- INRIA 的类别数最多，随机 pair 中约 98.97% 都是无关负样本。
- SSMH 的软语义监督在 INRIA 上几乎退化为极稀疏的二值监督，不能充分发挥“多标签复杂语义”的设计优势。

### 可能原因

1. **标签图平滑在单标签 INRIA 上可能是负贡献。**

   `semantic_similarity.py` 中的标签图来自类别共现矩阵。INRIA 每个样本只有一个标签，类别之间几乎没有共现；同时 `build_label_cooccurrence` 会把对角线置零。因此图传播后的标签会被削弱，而不是补充有用语义。

   在默认 `--use_label_graph True --graph_alpha 0.2` 下，单标签样本的同类相似度不会保持为理想的 1.0，而会被图平滑稀释。这会让同类正样本的目标相似度下降，削弱检索训练信号。

2. **当前 soft-margin rank loss 仍以“实例配对”为核心，而不是“同类多正样本”为核心。**

   `soft_margin_rank_loss` 使用 diagonal image-text pair 作为 paired positive，然后对所有 off-diagonal pair 做 margin 约束。对于 INRIA 这种类别检索任务，off-diagonal 中其实包含同类正样本，但当前 loss 只给它们较小 margin，并没有把它们显式作为正样本拉近。

   这会造成一个问题：模型努力让原始配对样本最相似，但对同类非配对样本的召回推动不足。mAP 评价却把同类样本都当作相关样本，所以指标会吃亏。

3. **soft pair MSE 被大量负样本主导。**

   INRIA 的零重叠 pair 比例约 98.97%，`semantic_pair_preserving_loss` 会让绝大多数 cross-modal pair 的预测相似度靠近 0。这个目标会压制全局相似度，容易牺牲少量同类正样本的排序。

4. **prototype loss 在单标签 100 类场景下可能过强。**

   当前 `composite_prototype_loss` 对单标签数据会退化为“每个样本靠近所属类别原型”。INRIA 每类平均约 90 个训练样本，类别多且视觉/文本分布可能复杂。训练早期的原型由未充分训练的特征估计，若 `--prototype_weight 1.0` 太强，可能过早把特征压到不稳定原型附近。

5. **NIRNL 更适配当前这批单标签噪声检索数据。**

   NIRNL 原本就是为 noisy label cross-modal retrieval 设计的，含 pure/hard/noisy 样本划分、邻居软标签和类中心细化。INRIA 当前标签结构更接近“大类别数单标签检索”，因此 NIRNL 的机制反而更贴合。

6. **SSMH 的优势需要真正多标签数据集来体现。**

   当前 SSMH 的创新点是多标签软语义、标签图、组合语义原型和哈希空间语义保持。但 INRIA 不是多标签，且类别共现关系几乎不存在，所以方法优势没有充分发挥。

### 下一步验证实验

优先在 INRIA 上做小规模定位，不要立刻扩大噪声比例：

| 优先级 | 实验 | 目的 | 命令参数 |
|---:|---|---|---|
| 1 | 关闭标签图 | 验证图平滑是否负贡献 | `--use_label_graph False` |
| 2 | 降低 soft pair 权重 | 减少大量负 pair 对 MSE 的主导 | `--soft_pair_weight 0.2` |
| 3 | 去掉 soft pair | 验证 pair MSE 是否主要伤害 MAP | `--soft_pair_weight 0` |
| 4 | 降低原型权重 | 减少不稳定类别原型约束 | `--prototype_weight 0.1` |
| 5 | 去掉原型 | 验证原型模块在 INRIA 是否负贡献 | `--prototype_weight 0` |
| 6 | 关闭图 + 降低 pair/prototype | 组合修正 | `--use_label_graph False --soft_pair_weight 0.2 --prototype_weight 0.1` |

建议第一条立即运行：

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --method ssmh \
  --dataset INRIA-Websearch \
  --use_label_graph False \
  --data_root /home/liuyizhi/NIRNL-AAAI26/Clean_idx \
  --noise_root /home/liuyizhi/nirnl-ssmh/noisy \
  --logging ablation_inria_ssmh_no_graph
```

判断标准：

- 如果关闭标签图后 Avg MAP 明显上升，说明单标签数据上图传播确实负贡献。
- 如果仍落后，则优先继续跑 `--soft_pair_weight 0.2` 和 `--prototype_weight 0.1`。
- 如果组合修正仍低于 NIRNL，说明当前 SSMH 主体需要改 loss：把同类 off-diagonal pair 显式作为 positives，而不是只依赖 diagonal paired positive。

## 当前下一步

主结果矩阵已经完成。下一步先定位 INRIA 上 SSMH 落后的原因，优先关闭标签图：

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --method ssmh \
  --dataset INRIA-Websearch \
  --use_label_graph False \
  --data_root /home/liuyizhi/NIRNL-AAAI26/Clean_idx \
  --noise_root /home/liuyizhi/nirnl-ssmh/noisy \
  --logging ablation_inria_ssmh_no_graph
```

若仍落后，继续跑：

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --method ssmh \
  --dataset INRIA-Websearch \
  --use_label_graph False \
  --soft_pair_weight 0.2 \
  --prototype_weight 0.1 \
  --data_root /home/liuyizhi/NIRNL-AAAI26/Clean_idx \
  --noise_root /home/liuyizhi/nirnl-ssmh/noisy \
  --logging ablation_inria_ssmh_no_graph_pair02_proto01
```

当前主结果观察：

- `wiki`：SSMH 略高于 NIRNL，但差距很小。
- `xmedia`：SSMH 小幅稳定领先 NIRNL。
- `INRIA-Websearch`：SSMH 明显落后 NIRNL，需要先做消融定位。
