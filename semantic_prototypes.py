import torch
import torch.nn.functional as F


def compute_class_prototypes(image_features, text_features, labels, eps=1e-8):
    """Compute one L2-normalized prototype per class from both modalities."""
    labels = labels.float()
    features = torch.cat([image_features, text_features], dim=0)
    repeated_labels = torch.cat([labels, labels], dim=0)
    class_mass = repeated_labels.sum(dim=0).clamp_min(eps).unsqueeze(1)
    prototypes = repeated_labels.t().mm(features) / class_mass
    return F.normalize(prototypes, p=2, dim=1)


def composite_prototypes(labels, class_prototypes, eps=1e-8):
    """Build one composite semantic prototype for each multi-label instance."""
    labels = labels.float()
    weights = labels / labels.sum(dim=1, keepdim=True).clamp_min(eps)
    composites = weights.mm(class_prototypes)
    return F.normalize(composites, p=2, dim=1)


def composite_prototype_loss(image_features, text_features, labels, class_prototypes):
    """Align each modality to its sample-specific composite prototype."""
    targets = composite_prototypes(labels, class_prototypes)
    image_loss = 1.0 - (image_features * targets).sum(dim=1)
    text_loss = 1.0 - (text_features * targets).sum(dim=1)
    return (image_loss + text_loss).mean()


def inter_prototype_structure_loss(class_prototypes, label_graph=None):
    """Optionally keep class prototype similarities close to a class graph."""
    if label_graph is None:
        gram = class_prototypes.mm(class_prototypes.t())
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return ((gram - eye) ** 2).mean()

    target = label_graph.float().to(class_prototypes.device)
    target = target.clamp(0.0, 1.0)
    pred = (class_prototypes.mm(class_prototypes.t()) + 1.0) / 2.0
    return F.mse_loss(pred, target)
