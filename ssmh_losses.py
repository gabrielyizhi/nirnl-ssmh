import torch
import torch.nn.functional as F

from semantic_similarity import semantic_similarity
from semantic_prototypes import composite_prototype_loss


def adaptive_margin(soft_similarity, margin_min=0.1, margin_max=0.7):
    """Map semantic similarity to a pair-specific margin."""
    return margin_min + (1.0 - soft_similarity).clamp(0.0, 1.0) * (margin_max - margin_min)


def soft_margin_rank_loss(image_features, text_features, labels, args, label_graph=None):
    """Cross-modal ranking loss with semantic adaptive margins."""
    soft_sim = semantic_similarity(
        labels,
        mode=args.semantic_sim,
        label_graph=label_graph,
        graph_alpha=args.graph_alpha,
        jaccard_weight=args.jaccard_weight,
        overlap_weight=args.overlap_weight,
        cosine_weight=args.cosine_weight,
    )
    margins = adaptive_margin(soft_sim, args.margin_min, args.margin_max)
    sim12 = image_features.mm(text_features.t())
    sim21 = text_features.mm(image_features.t())

    paired12 = torch.diag(sim12).view(-1, 1)
    paired21 = torch.diag(sim21).view(-1, 1)
    mask = 1.0 - torch.eye(sim12.shape[0], device=sim12.device, dtype=sim12.dtype)

    loss12 = F.relu(sim12 - paired12 + margins) * mask
    loss21 = F.relu(sim21 - paired21 + margins) * mask
    denom = mask.sum().clamp_min(1.0)
    return (loss12.sum() + loss21.sum()) / denom


def semantic_pair_preserving_loss(image_features, text_features, labels, args, label_graph=None):
    """Match cross-modal pair similarities to continuous semantic similarities."""
    soft_sim = semantic_similarity(
        labels,
        mode=args.semantic_sim,
        label_graph=label_graph,
        graph_alpha=args.graph_alpha,
        jaccard_weight=args.jaccard_weight,
        overlap_weight=args.overlap_weight,
        cosine_weight=args.cosine_weight,
    )
    pred12 = (image_features.mm(text_features.t()) + 1.0) / 2.0
    pred21 = (text_features.mm(image_features.t()) + 1.0) / 2.0
    return 0.5 * (F.mse_loss(pred12, soft_sim) + F.mse_loss(pred21, soft_sim))


def quantization_loss(features):
    """Encourage L2-normalized features to approach normalized binary vertices."""
    target_abs = 1.0 / (features.shape[1] ** 0.5)
    return ((features.abs() - target_abs) ** 2).mean()


def bit_balance_loss(features):
    """Encourage each hash bit to be balanced across a mini-batch."""
    return features.mean(dim=0).pow(2).mean()


def bit_decorrelation_loss(features):
    """Reduce redundant hash bits by penalizing off-diagonal correlations."""
    if features.shape[0] <= 1:
        return features.new_tensor(0.0)
    centered = features - features.mean(dim=0, keepdim=True)
    corr = centered.t().mm(centered) / (features.shape[0] - 1)
    eye = torch.eye(corr.shape[0], device=features.device, dtype=features.dtype)
    return ((corr * (1.0 - eye)) ** 2).mean()


def hash_space_regularization(image_features, text_features):
    all_features = torch.cat([image_features, text_features], dim=0)
    return {
        "quant": quantization_loss(all_features),
        "balance": bit_balance_loss(all_features),
        "decorrelation": bit_decorrelation_loss(all_features),
    }


def ssmh_loss(image_features, text_features, labels, class_prototypes, args, label_graph=None):
    """Full SSMH objective for one mini-batch."""
    rank = soft_margin_rank_loss(image_features, text_features, labels, args, label_graph)
    pair = semantic_pair_preserving_loss(image_features, text_features, labels, args, label_graph)
    proto = composite_prototype_loss(image_features, text_features, labels, class_prototypes)
    regs = hash_space_regularization(image_features, text_features)
    total = (
        args.alpha * rank
        + args.soft_pair_weight * pair
        + args.prototype_weight * proto
        + args.quant_weight * regs["quant"]
        + args.balance_weight * regs["balance"]
        + args.decorrelation_weight * regs["decorrelation"]
    )
    parts = {
        "rank": rank.detach(),
        "pair": pair.detach(),
        "prototype": proto.detach(),
        "quant": regs["quant"].detach(),
        "balance": regs["balance"].detach(),
        "decorrelation": regs["decorrelation"].detach(),
    }
    return total, parts
