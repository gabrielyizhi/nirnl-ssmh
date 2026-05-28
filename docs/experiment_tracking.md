# SSMH 实验计划与结果追踪

更新时间：2026-05-27

本文档用于追踪 `nirnl-ssmh` 后续所有实验。每次实验跑完后，需要在“结果记录”中追加一行，并在“阶段状态”中更新完成情况。

## 当前结论

- 服务器当前可用数据集：`wiki`、`xmedia`、`INRIA-Websearch`。
- 代码默认数据集是 `nuswide`，但服务器当前缺少 `nus_wide_deep_doc2vec-corr-ae.h5py`，所以不能直接用默认 `dataset=nuswide` 跑 AAAI 默认流程。
- 当前不缺启动实验的数据集：先用 `wiki`、`xmedia`、`INRIA-Websearch` 可以完成主结果、噪声鲁棒性和消融验证。
- 若要做更标准、更有说服力的大规模多标签跨模态哈希实验，需要补 `NUS-WIDE` 的预处理特征文件，或者新增数据预处理流程。
- 服务器旧项目中存在 `nuswide` / `xmedianet` 的历史结果文件，但在 `/home/liuyizhi` 下没有找到当前代码期望的预处理数据文件。
- 论文当前最应该先做：在 `wiki` 上跑 `SSMH`，和已经跑通的 `NIRNL` 形成第一组直接对比。

## 服务器数据集状态

数据根目录：

```bash
/home/liuyizhi/NIRNL-AAAI26/Clean_idx
```

| 数据集参数名 | 代码期望文件 | 服务器状态 | 备注 |
|---|---|---|---|
| `wiki` | `wiki.mat` | 可用 | 已跑通 NIRNL 100 epoch |
| `xmedia` | `XMediaFeatures.mat` | 可用 | 待跑 |
| `INRIA-Websearch` | `INRIA-Websearch.mat` | 可用 | 待跑 |
| `nuswide` | `nus_wide_deep_doc2vec-corr-ae.h5py` | 缺失 | AAAI 默认 dataset，但当前无法直接跑 |
| `xmedianet` | `xmedianet_deep_doc2vec_data.h5py` + `XMediaNet5View_Doc2Vec.mat` | 缺失 | 可作为后续扩展数据集 |

只读搜索结果：

- 未找到 `/home/liuyizhi/**/nus_wide_deep_doc2vec-corr-ae.h5py`。
- 未找到 `/home/liuyizhi/**/xmedianet_deep_doc2vec_data.h5py`。
- 未找到 `/home/liuyizhi/**/XMediaNet5View_Doc2Vec.mat`。
- 找到了旧实验输出，例如 `/home/liuyizhi/NIRNL-AAAI26/Avg_MAP_data/nuswide_Avg_MAP_0.2_0_.mat` 和 `/home/liuyizhi/NIRNL-AAAI26/logging/nuswide.log`，说明历史上可能在别的路径或别的机器上跑过 NUS-WIDE。

## 阶段状态

| 阶段 | 目标 | 状态 | 下一步 |
|---|---|---|---|
| 环境验证 | 服务器连接、GPU、依赖、代码入口跑通 | 完成 | 无 |
| Baseline 验证 | WIKI + NIRNL + 默认 AAAI 超参 | 完成 | 作为第一条 baseline |
| 主方法验证 | WIKI + SSMH + 同等超参 | 待做 | 立即执行 |
| 主结果表 | 3 个现有数据集上比较 NIRNL 和 SSMH | 待做 | WIKI 后跑 XMedia、INRIA |
| 噪声鲁棒性 | 多噪声比例比较 NIRNL 和 SSMH | 待做 | 先 WIKI，后扩展 |
| 消融实验 | 验证 SSMH 各模块贡献 | 待做 | 先 WIKI r=0.2/r=0.4 |
| 多 seed | 统计稳定性 `mean ± std` | 待做 | 主趋势稳定后补 |
| NUS-WIDE | 获取或构建 NUS-WIDE 预处理特征 | 待做 | 可并行准备，但不是当前阻塞项 |

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
| 5 | `INRIA-Websearch` | `nirnl` | 0.2 | 1 | 待跑 |
| 6 | `INRIA-Websearch` | `ssmh` | 0.2 | 1 | 待跑 |

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

### 当前问题

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

因此，仅下载原始 NUS-WIDE 标注包还不能直接跑当前代码。可选方案：

1. 找到原论文/旧项目使用的预处理 `.h5py` 文件，放到 `/home/liuyizhi/NIRNL-AAAI26/Clean_idx`。
2. 下载 NUS-WIDE 原始标签与图像，再新增预处理脚本，抽取图像深度特征和文本/doc2vec 特征，生成代码期望的 `.h5py`。
3. 暂时不使用 NUS-WIDE，先用已有 3 个数据集完成方法验证和论文初步结果。

### 建议

短期不要让 NUS-WIDE 阻塞实验。当前先跑完 `wiki/xmedia/INRIA-Websearch` 的主结果和 WIKI 消融；同时并行寻找或构建 NUS-WIDE 预处理特征。

## 结果记录

每次实验跑完后，在这里新增一行。

| 日期 | 数据集 | 方法 | 噪声比例 | seed | 额外参数 | I2T MAP | T2I MAP | Avg MAP | I2T nDCG@100 | T2I nDCG@100 | Hamming Spearman | 日志名 | 状态 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-05-27 | `wiki` | `nirnl` | 0.2 | 1 | AAAI default | 0.523821 | 0.486570 | 0.505195 | 0.655180 | 0.670087 | 0.313426 | `aaaidefault_wiki_nirnl` | 完成 |
| 2026-05-28 | `wiki` | `ssmh` | 0.2 | 1 | AAAI default + SSMH default | 0.530816 | 0.479868 | 0.505342 | 0.649357 | 0.669582 | 0.321967 | `aaaidefault_wiki_ssmh` | 完成 |
| 2026-05-28 | `xmedia` | `nirnl` | 0.2 | 1 | AAAI default | 0.918511 | 0.916813 | 0.917662 | 0.941751 | 0.958052 | 0.357104 | `aaaidefault_xmedia_nirnl` | 完成 |
| 2026-05-28 | `xmedia` | `ssmh` | 0.2 | 1 | AAAI default + SSMH default | 0.920312 | 0.917910 | 0.919111 | 0.945662 | 0.958071 | 0.358795 | `aaaidefault_xmedia_ssmh` | 完成 |

## 当前下一步

WIKI 第一组主结果已经完成，下一步转向 `xmedia`，先跑 baseline，再跑 SSMH：

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --method nirnl \
  --dataset xmedia \
  --data_root /home/liuyizhi/NIRNL-AAAI26/Clean_idx \
  --noise_root /home/liuyizhi/nirnl-ssmh/noisy \
  --logging aaaidefault_xmedia_nirnl
```

随后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --method ssmh \
  --dataset xmedia \
  --data_root /home/liuyizhi/NIRNL-AAAI26/Clean_idx \
  --noise_root /home/liuyizhi/nirnl-ssmh/noisy \
  --logging aaaidefault_xmedia_ssmh
```

WIKI 当前观察：

- `SSMH Avg MAP = 0.505342`，略高于 `NIRNL Avg MAP = 0.505195`。
- `SSMH I2T MAP` 更高，但 `T2I MAP` 和 nDCG 略低。
- `SSMH Hamming Spearman = 0.321967`，高于 `NIRNL = 0.313426`，说明语义顺序保持指标有改善。
- 当前结论：WIKI 上主方法没有明显拉开差距，需要继续看 `xmedia` 和 `INRIA-Websearch`，并准备后续权重调参与消融定位。
