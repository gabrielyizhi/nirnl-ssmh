import numpy as np
import torch


def _safe_divide(numerator, denominator, eps=1e-8):
    return numerator / denominator.clamp_min(eps)


def label_jaccard_similarity(labels_a, labels_b=None, eps=1e-8):
    """Compute multi-label Jaccard similarity for torch tensors."""
    if labels_b is None:
        labels_b = labels_a
    labels_a = labels_a.float()
    labels_b = labels_b.float()
    intersection = labels_a.mm(labels_b.t())
    count_a = labels_a.sum(dim=1, keepdim=True)
    count_b = labels_b.sum(dim=1, keepdim=True).t()
    union = count_a + count_b - intersection
    return _safe_divide(intersection, union, eps=eps).clamp(0.0, 1.0)


def label_overlap_similarity(labels_a, labels_b=None, eps=1e-8):
    """Intersection over the smaller label set; useful for inclusion relations."""
    if labels_b is None:
        labels_b = labels_a
    labels_a = labels_a.float()
    labels_b = labels_b.float()
    intersection = labels_a.mm(labels_b.t())
    count_a = labels_a.sum(dim=1, keepdim=True)
    count_b = labels_b.sum(dim=1, keepdim=True).t()
    smaller = torch.minimum(count_a, count_b)
    return _safe_divide(intersection, smaller, eps=eps).clamp(0.0, 1.0)


def label_cosine_similarity(labels_a, labels_b=None, eps=1e-8):
    """Cosine similarity over multi-hot labels."""
    if labels_b is None:
        labels_b = labels_a
    labels_a = labels_a.float()
    labels_b = labels_b.float()
    dot = labels_a.mm(labels_b.t())
    norm_a = labels_a.norm(p=2, dim=1, keepdim=True)
    norm_b = labels_b.norm(p=2, dim=1, keepdim=True).t()
    return _safe_divide(dot, norm_a.mm(norm_b), eps=eps).clamp(0.0, 1.0)


def build_label_cooccurrence(labels, normalize=True, eps=1e-8):
    """Build a category co-occurrence matrix from numpy or torch labels."""
    is_numpy = isinstance(labels, np.ndarray)
    label_tensor = torch.from_numpy(labels).float() if is_numpy else labels.float()
    cooc = label_tensor.t().mm(label_tensor)
    cooc.fill_diagonal_(0)
    if normalize:
        degree = cooc.sum(dim=1, keepdim=True)
        cooc = cooc / degree.clamp_min(eps)
    return cooc.cpu().numpy() if is_numpy else cooc


def graph_smoothed_labels(labels, label_graph, alpha=0.2):
    """Smooth labels through a class graph while keeping original labels dominant."""
    labels = labels.float()
    graph = label_graph.float()
    propagated = labels.mm(graph)
    smoothed = (1.0 - alpha) * labels + alpha * propagated
    return smoothed.clamp_min(0.0)


def semantic_similarity(labels_a, labels_b=None, mode="hybrid", label_graph=None,
                        graph_alpha=0.2, jaccard_weight=0.5,
                        overlap_weight=0.3, cosine_weight=0.2):
    """Return continuous semantic similarity in [0, 1].

    `hybrid` combines Jaccard, overlap, and cosine similarities. If a label
    graph is supplied, labels are smoothed before the similarity is computed.
    """
    if labels_b is None:
        labels_b = labels_a
    labels_a = labels_a.float()
    labels_b = labels_b.float()

    if label_graph is not None and graph_alpha > 0:
        labels_a = graph_smoothed_labels(labels_a, label_graph, graph_alpha)
        labels_b = graph_smoothed_labels(labels_b, label_graph, graph_alpha)

    if mode == "jaccard":
        return label_jaccard_similarity(labels_a, labels_b)
    if mode == "overlap":
        return label_overlap_similarity(labels_a, labels_b)
    if mode == "cosine":
        return label_cosine_similarity(labels_a, labels_b)
    if mode != "hybrid":
        raise ValueError("Unknown semantic similarity mode: {}".format(mode))

    total = jaccard_weight + overlap_weight + cosine_weight
    if total <= 0:
        raise ValueError("At least one semantic similarity weight must be positive.")
    sim = (
        jaccard_weight * label_jaccard_similarity(labels_a, labels_b)
        + overlap_weight * label_overlap_similarity(labels_a, labels_b)
        + cosine_weight * label_cosine_similarity(labels_a, labels_b)
    ) / total
    return sim.clamp(0.0, 1.0)
