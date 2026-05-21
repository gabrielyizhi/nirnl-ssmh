from __future__ import division
from __future__ import print_function

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import (calc_ndcg_multilabel, calc_weighted_map_multilabel,
                      fx_calc_map_multilabel)
from load_data import CustomDataSet
from semantic_prototypes import compute_class_prototypes
from semantic_similarity import build_label_cooccurrence
from ssmh_losses import ssmh_loss


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_tensor(array, device):
    return torch.tensor(array, dtype=torch.float32, device=device)


def extract_features(model, images, texts, batch_size=512):
    device = _device()
    model.eval()
    image_features, text_features = [], []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            end = start + batch_size
            img = _as_tensor(images[start:end], device)
            txt = _as_tensor(texts[start:end], device)
            img_fea, txt_fea = model(img, txt)
            image_features.append(img_fea.cpu())
            text_features.append(txt_fea.cpu())
    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0)


def build_epoch_semantics(model, img_train, txt_train, labels, args):
    device = _device()
    image_features, text_features = extract_features(
        model, img_train, txt_train, batch_size=args.eval_batch_size
    )
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    prototypes = compute_class_prototypes(image_features, text_features, label_tensor)
    label_graph = None
    if args.use_label_graph:
        label_graph = build_label_cooccurrence(label_tensor, normalize=True)
        label_graph = label_graph.to(device)
    return prototypes.to(device), label_graph


def evaluate_model(model, img_data, txt_data, labels, args):
    image_features, text_features = extract_features(
        model, img_data, txt_data, batch_size=args.eval_batch_size
    )
    image_np = image_features.numpy()
    text_np = text_features.numpy()
    map_i2t = fx_calc_map_multilabel(image_np, text_np, labels, metric="cosine")
    map_t2i = fx_calc_map_multilabel(text_np, image_np, labels, metric="cosine")
    ndcg_i2t = calc_ndcg_multilabel(image_np, text_np, labels, k=args.ndcg_k, metric="cosine")
    ndcg_t2i = calc_ndcg_multilabel(text_np, image_np, labels, k=args.ndcg_k, metric="cosine")
    wmap_i2t = calc_weighted_map_multilabel(image_np, text_np, labels, metric="cosine")
    wmap_t2i = calc_weighted_map_multilabel(text_np, image_np, labels, metric="cosine")
    return {
        "map_i2t": map_i2t,
        "map_t2i": map_t2i,
        "ndcg_i2t": ndcg_i2t,
        "ndcg_t2i": ndcg_t2i,
        "wmap_i2t": wmap_i2t,
        "wmap_t2i": wmap_t2i,
        "avg_map": (map_i2t + map_t2i) / 2.0,
    }


def train_one_epoch(model, train_loader, optimizer, class_prototypes, label_graph, args):
    device = _device()
    model.train()
    running = {}
    total_batches = 0
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        imgs = imgs.to(device).float()
        txts = txts.to(device).float()
        labels = labels_ori.to(device).float() if args.ssmh_use_clean_labels else labels_noisy.to(device).float()

        optimizer.zero_grad()
        image_features, text_features = model(imgs, txts)
        loss, parts = ssmh_loss(
            image_features, text_features, labels, class_prototypes, args, label_graph=label_graph
        )
        loss.backward()
        optimizer.step()

        running["total"] = running.get("total", 0.0) + float(loss.detach().cpu())
        for key, value in parts.items():
            running[key] = running.get(key, 0.0) + float(value.cpu())
        total_batches += 1

    return {key: value / max(total_batches, 1) for key, value in running.items()}


def train_model_ssmh(model, input_data_par, optimizer, args):
    img_train = input_data_par["img_train"]
    txt_train = input_data_par["text_train"]
    label_train_ori = input_data_par["label_train_ori"]
    label_train_noisy = input_data_par["label_train_noisy"]
    img_valid = input_data_par["img_valid"]
    txt_valid = input_data_par["text_valid"]
    label_valid = input_data_par["label_valid"]

    train_dataset = CustomDataSet(img_train, txt_train, label_train_noisy, label_train_ori)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    since = time.time()
    best_score = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    semantic_labels = label_train_ori if args.ssmh_use_clean_labels else label_train_noisy
    for epoch in range(args.MAX_EPOCH):
        print("SSMH Epoch {}/{}".format(epoch + 1, args.MAX_EPOCH))
        print("-" * 20)
        class_prototypes, label_graph = build_epoch_semantics(
            model, img_train, txt_train, semantic_labels, args
        )
        losses = train_one_epoch(
            model, train_loader, optimizer, class_prototypes, label_graph, args
        )
        metrics = evaluate_model(model, img_valid, txt_valid, label_valid, args)
        history.append({**losses, **metrics})

        print(
            "Loss: {total:.4f} Rank: {rank:.4f} Pair: {pair:.4f} "
            "Proto: {prototype:.4f} mAP: {avg_map:.4f} "
            "nDCG: {ndcg:.4f} wMAP: {wmap:.4f}".format(
                total=losses.get("total", 0.0),
                rank=losses.get("rank", 0.0),
                pair=losses.get("pair", 0.0),
                prototype=losses.get("prototype", 0.0),
                avg_map=metrics["avg_map"],
                ndcg=(metrics["ndcg_i2t"] + metrics["ndcg_t2i"]) / 2.0,
                wmap=(metrics["wmap_i2t"] + metrics["wmap_t2i"]) / 2.0,
            )
        )

        if metrics["avg_map"] > best_score:
            best_score = metrics["avg_map"]
            best_model_wts = copy.deepcopy(model.state_dict())

    elapsed = time.time() - since
    print("SSMH training complete in {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))
    print("Best validation average MAP: {:.4f}".format(best_score))
    model.load_state_dict(best_model_wts)
    return model, history
