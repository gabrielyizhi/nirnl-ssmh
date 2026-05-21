import numpy as np
import scipy
import scipy.spatial
import scipy.stats
import torch


def _graded_relevance(query_label, database_labels):
    intersection = np.dot(database_labels, query_label)
    query_count = np.sum(query_label)
    database_count = np.sum(database_labels, axis=1)
    union = query_count + database_count - intersection
    rel = np.divide(
        intersection,
        np.maximum(union, 1e-8),
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    return rel


def fx_calc_map_label(image, text, label, k = 0, dist_method='COS'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

def fx_calc_map_multilabel(image, text, label, k = 0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(image, text, metric)
    ord = dist.argsort()
    
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    for i in range(dist.shape[0]):
        order = ord[i].reshape(-1)[0: dist.shape[0]]

        tmp_label = (np.dot(label[order], label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)


def calc_weighted_map_multilabel(image, text, label, k=0, metric='cosine'):
    """Weighted mAP with Jaccard label overlap as graded relevance."""
    dist = scipy.spatial.distance.cdist(image, text, metric)
    order = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    scores = []
    for i in range(numcases):
        ranking = order[i][:k]
        rel = _graded_relevance(label[i], label[ranking])
        if rel.sum() <= 0:
            scores.append(0.0)
            continue
        precision = np.cumsum(rel) / np.arange(1.0, rel.shape[0] + 1.0)
        scores.append(np.dot(rel, precision) / rel.sum())
    return float(np.mean(scores)) if scores else 0.0


def calc_ndcg_multilabel(image, text, label, k=100, metric='cosine'):
    """nDCG using Jaccard label overlap as graded relevance."""
    dist = scipy.spatial.distance.cdist(image, text, metric)
    order = dist.argsort()
    numcases = dist.shape[0]
    k = min(k if k else numcases, numcases)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    scores = []
    for i in range(numcases):
        ranking = order[i][:k]
        rel = _graded_relevance(label[i], label[ranking])
        dcg = np.dot(rel, discounts[: rel.shape[0]])
        all_rel = _graded_relevance(label[i], label)
        ideal = np.sort(all_rel)[::-1][:k]
        idcg = np.dot(ideal, discounts[: ideal.shape[0]])
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def calc_hamming_semantic_correlation(image, text, label, sample_size=5000):
    """Spearman correlation between Hamming distance and semantic similarity."""
    img_hash = np.sign(image)
    txt_hash = np.sign(text)
    rng = np.random.default_rng(0)
    n = min(image.shape[0], text.shape[0])
    if n == 0:
        return 0.0
    pair_count = min(sample_size, n * n)
    query_idx = rng.integers(0, n, size=pair_count)
    db_idx = rng.integers(0, n, size=pair_count)
    hamming = np.mean(img_hash[query_idx] != txt_hash[db_idx], axis=1)
    semantic = np.array([
        _graded_relevance(label[q], label[d:d + 1])[0]
        for q, d in zip(query_idx, db_idx)
    ])
    if np.std(hamming) < 1e-8 or np.std(semantic) < 1e-8:
        return 0.0
    return float(scipy.stats.spearmanr(-hamming, semantic).correlation)

def predict(model, data, index = 0, batch_size=32):
    batch_count = int(np.ceil(data.shape[0] / float(batch_size)))
    results= []
    with torch.no_grad():
        for i in range(batch_count):
            if torch.is_tensor(data):
                batch = data[i * batch_size: (i + 1) * batch_size]
            else:
                batch = (torch.tensor(data[i * batch_size: (i + 1) * batch_size])).cuda()
            results.append(model(batch))
        batch = (torch.tensor(data[batch_count * batch_size:])).cuda()
        results.append(model(batch))
    return torch.concat(results)
