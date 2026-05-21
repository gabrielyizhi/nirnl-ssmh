# SSMH module plan

## Current NIRNL code understanding

This repository is the implementation base for the NIRNL paper,
`Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval`.
The current pipeline is:

1. `main.py` parses experiment arguments, loads feature-level image/text data,
   builds `IDCM_NN`, and calls `train_model_synchronous`.
2. `load_data.py` loads prepared `.mat` / `.h5py` features, converts single-label
   labels to one-hot form when needed, injects synthetic noisy labels, and returns
   numpy arrays plus metadata.
3. `model.py` defines two MLP encoders, `ImgNN` and `TextNN`, wrapped by
   `IDCM_NN`. Both encoders output L2-normalized continuous representations.
4. `train_model_knn_v2.py` contains the main NIRNL logic:
   - `rank_loss` is the current fixed-margin cross-modal margin preserving loss.
   - `get_barycenters` builds class barycenters from image/text features.
   - `get_soft_labels` averages labels from top-k neighbors.
   - `divide_sample` partitions samples into pure, hard, and noisy subsets.
   - `train_pure`, `train_hard`, and `train_noisy` apply subset-specific losses.
5. `evaluate.py` computes standard mAP with binary relevance defined by whether
   two samples share at least one label.

The key reusable parts for the next paper are the data interface, dual-encoder
feature mapping, cross-modal margin idea, barycenter/prototype mechanism, and
neighbor-aware soft label generation.

## Target research direction

The next method should focus on multi-label complex semantics rather than noisy
label reliability. A working method name is:

**Structure-aware Soft-Margin Cross-Modal Hashing (SSMH)**

The new method should answer:

> How can multi-label supervision be transformed from hard binary pairwise
> labels into structured, continuous, rank-preserving semantic supervision, and
> how can this supervision be preserved in compact Hamming space?

## Modules to add

### 1. Semantic relation builder

Add a module that constructs a continuous soft similarity matrix from multi-label
annotations.

Suggested file:

- `semantic_similarity.py`

Responsibilities:

- Compute Jaccard / overlap / cosine similarity from multi-hot labels.
- Optionally build a category co-occurrence graph from training labels.
- Optionally smooth label similarities through graph propagation.
- Return batch-level or full-dataset soft similarity values in `[0, 1]`.

Initial implementation should start simple:

- `label_jaccard_similarity(labels_a, labels_b)`
- `label_overlap_similarity(labels_a, labels_b)`
- `build_label_cooccurrence(labels)`
- `graph_smoothed_label_similarity(labels, graph, alpha)`

### 2. Soft-margin cross-modal loss

Replace or extend the current fixed-margin `rank_loss` with a semantic adaptive
margin loss.

Suggested file:

- `ssmh_losses.py`

Responsibilities:

- Map soft similarity `s_ij` to a pair-specific margin `m_ij`.
- Pull highly related cross-modal pairs closer.
- Keep partially related pairs at intermediate distances.
- Push weakly related or unrelated pairs farther away.

The current `rank_loss(features1, features2, margin)` in
`train_model_knn_v2.py` is the direct insertion point.

Candidate formula:

```text
m_ij = m_min + (1 - s_ij) * (m_max - m_min)
```

### 3. Composite semantic prototype module

Generalize the current class barycenter mechanism to multi-label composite
semantic prototypes.

Suggested file:

- `semantic_prototypes.py`

Responsibilities:

- Learn or compute one proxy/prototype per class.
- Construct a sample-specific composite prototype by weighted combination of
  class prototypes.
- Align image/text features with their composite prototype.
- Preserve inter-prototype structure.

The current `get_barycenters` function is the nearest existing implementation
reference.

### 4. Hash-space preservation losses

Add losses that explicitly preserve semantic ordering after binarization.

Suggested additions:

- Quantization loss.
- Bit balance loss.
- Bit decorrelation loss.
- Optional weighted Hamming distance regularization.

These should be implemented as independent functions first, then combined in
the training loop through configurable weights.

### 5. Fine-grained evaluation metrics

Extend `evaluate.py`, because current mAP only uses binary relevance:

```python
tmp_label = (np.dot(label[order], label[i]) > 0)
```

This is insufficient for the new paper because it cannot distinguish strong,
partial, and weak semantic matches.

Add:

- `calc_ndcg_multilabel`
- `calc_weighted_map_multilabel`
- `calc_hamming_semantic_correlation`
- distance distribution by label-overlap level

These metrics will directly support the paper claim that SSMH preserves
fine-grained multi-label semantic order.

### 6. Dataset path and experiment config cleanup

The current dataset paths are hard-coded under `/home/qinyang/...`. Before major
experiments, add a small config layer.

Suggested files:

- `configs/datasets.yaml`
- `configs/ssmh_default.yaml`

Suggested changes:

- Add `--data_root`.
- Add dataset-specific relative filenames.
- Keep legacy defaults available for reproducibility.

## Proposed implementation order

1. Add fine-grained metrics in `evaluate.py`.
2. Add `semantic_similarity.py` and verify soft similarity statistics on labels.
3. Add `ssmh_losses.py` with soft-margin loss.
4. Add a new training entry, e.g. `train_model_ssmh.py`, instead of rewriting
   `train_model_knn_v2.py` immediately.
5. Add composite prototype logic.
6. Add quantization / bit balance / bit decorrelation losses.
7. Add ablation switches in `main.py`.
8. Run MIRFlickr-25K or the smallest available multi-label dataset first.

## Experiment matrix

Main datasets:

- MIRFlickr-25K
- NUS-WIDE
- MS-COCO
- Optional: IAPR-TC12

Metrics:

- mAP, P@K, PR curve
- nDCG@K
- weighted mAP
- Spearman / Kendall correlation between soft similarity and Hamming distance
- average Hamming distance by shared-label count

Ablations:

- hard binary similarity only
- Jaccard-only soft similarity
- without graph smoothing
- without soft margin
- without composite prototype
- without hash-space preservation losses

## Notes for Git workflow

- `main` should keep the exact NIRNL base import.
- `codex/ssmh-development` is the working branch for SSMH changes.
- Every functional change should be committed separately.
- Push only after a GitHub remote for this new repository is configured.
