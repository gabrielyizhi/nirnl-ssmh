import argparse
import logging
import os

import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim

from evaluate import (calc_hamming_semantic_correlation, calc_ndcg_multilabel,
                      calc_weighted_map_multilabel, fx_calc_map_multilabel)
from load_data import get_loader
from model import IDCM_NN
from to_seed import to_seed
from train_model_knn_v2 import train_model_synchronous
from train_model_ssmh import train_model_ssmh


def str2bool(value):
    return True if str(value).lower() == "true" else False


parser = argparse.ArgumentParser(description="NIRNL / SSMH cross-modal retrieval")

#########################
#### data parameters ####
#########################
parser.add_argument("--dataset", type=str, default="nuswide")
parser.add_argument("--data_root", type=str, default=None)
parser.add_argument("--noise_root", type=str, default=None)
parser.add_argument("--method", type=str, default="nirnl", choices=["nirnl", "ssmh"])
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--output_dim", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--MAX_EPOCH", type=int, default=100)
parser.add_argument("--GPU", type=int, default=0)
parser.add_argument("--logging", type=str, default=None)

#########################
#### NIRNL parameters ####
#########################
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--lamda", type=float, default=1)
parser.add_argument("--margin", type=float, default=0.7)
parser.add_argument("--lambd", default=5, type=float)
parser.add_argument("--barycenter_number", default=1, type=int)
parser.add_argument("--top_k", default=10, type=int)
parser.add_argument("--warm_up_epoch", type=int, default=0)
parser.add_argument("--noisy_ratio", type=float, default=0.2)
parser.add_argument("--noise_mode", type=str, default="sym")
parser.add_argument("--hard_weight", type=str2bool, default=True)
parser.add_argument("--noisy_train", type=str2bool, default=True)

#######################
#### SSMH settings ####
#######################
parser.add_argument("--semantic_sim", type=str, default="hybrid",
                    choices=["hybrid", "jaccard", "overlap", "cosine"])
parser.add_argument("--use_label_graph", type=str2bool, default=True)
parser.add_argument("--graph_alpha", type=float, default=0.2)
parser.add_argument("--jaccard_weight", type=float, default=0.5)
parser.add_argument("--overlap_weight", type=float, default=0.3)
parser.add_argument("--cosine_weight", type=float, default=0.2)
parser.add_argument("--margin_min", type=float, default=0.1)
parser.add_argument("--margin_max", type=float, default=0.7)
parser.add_argument("--soft_pair_weight", type=float, default=1.0)
parser.add_argument("--prototype_weight", type=float, default=1.0)
parser.add_argument("--quant_weight", type=float, default=0.05)
parser.add_argument("--balance_weight", type=float, default=0.01)
parser.add_argument("--decorrelation_weight", type=float, default=0.01)
parser.add_argument("--ssmh_use_clean_labels", type=str2bool, default=True)
parser.add_argument("--eval_batch_size", type=int, default=512)
parser.add_argument("--ndcg_k", type=int, default=100)


def setup_logger(args):
    os.makedirs("logging", exist_ok=True)
    logger_name = args.logging if args.logging else "{}_{}".format(args.method, args.dataset)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join("logging", logger_name + ".log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def evaluate_and_save_features(model, input_data_par, args, logger, history):
    dataset = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("...Evaluation on testing data...")
    with torch.no_grad():
        view1_feature, view2_feature = model(
            torch.tensor(input_data_par["img_test"]).float().to(device),
            torch.tensor(input_data_par["text_test"]).float().to(device),
        )
    label = input_data_par["label_test"]
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()

    os.makedirs("Avg_MAP_data", exist_ok=True)
    result_path = "Avg_MAP_data/{}_{}_Avg_MAP_{:.1f}_{}.mat".format(
        args.method, dataset, args.noisy_ratio, args.warm_up_epoch
    )
    if args.method == "ssmh":
        sio.savemat(
            result_path,
            {"history": np.array([str(item) for item in history], dtype=object),
             "max_epoch": args.MAX_EPOCH},
        )
    else:
        sio.savemat(result_path, {"map_list": history, "max_epoch": args.MAX_EPOCH})

    img_to_txt = fx_calc_map_multilabel(view1_feature, view2_feature, label, metric="cosine")
    txt_to_img = fx_calc_map_multilabel(view2_feature, view1_feature, label, metric="cosine")
    ndcg_i2t = calc_ndcg_multilabel(view1_feature, view2_feature, label, k=args.ndcg_k, metric="cosine")
    ndcg_t2i = calc_ndcg_multilabel(view2_feature, view1_feature, label, k=args.ndcg_k, metric="cosine")
    wmap_i2t = calc_weighted_map_multilabel(view1_feature, view2_feature, label, metric="cosine")
    wmap_t2i = calc_weighted_map_multilabel(view2_feature, view1_feature, label, metric="cosine")
    semantic_corr = calc_hamming_semantic_correlation(view1_feature, view2_feature, label)

    messages = [
        "...Image to Text MAP = {}".format(img_to_txt),
        "...Text to Image MAP = {}".format(txt_to_img),
        "...Average MAP = {}".format((img_to_txt + txt_to_img) / 2.0),
        "...Image to Text nDCG@{} = {}".format(args.ndcg_k, ndcg_i2t),
        "...Text to Image nDCG@{} = {}".format(args.ndcg_k, ndcg_t2i),
        "...Image to Text weighted MAP = {}".format(wmap_i2t),
        "...Text to Image weighted MAP = {}".format(wmap_t2i),
        "...Hamming semantic Spearman = {}".format(semantic_corr),
    ]
    for message in messages:
        print(message)
        logger.info(message)

    os.makedirs("features", exist_ok=True)
    train_label = np.argmax(input_data_par["label_train_ori"], axis=1)
    sio.savemat(
        "features/{}_0_train_ori.mat".format(dataset),
        {"train_fea": input_data_par["img_train"], "train_lab": train_label},
    )
    sio.savemat(
        "features/{}_1_train_ori.mat".format(dataset),
        {"train_fea": input_data_par["text_train"], "train_lab": train_label},
    )
    with torch.no_grad():
        train_img, train_txt = model(
            torch.tensor(input_data_par["img_train"]).float().to(device),
            torch.tensor(input_data_par["text_train"]).float().to(device),
        )
    sio.savemat(
        "features/{}_0_train.mat".format(dataset),
        {"train_fea": train_img.detach().cpu().numpy(), "train_lab": train_label},
    )
    sio.savemat(
        "features/{}_1_train.mat".format(dataset),
        {"train_fea": train_txt.detach().cpu().numpy(), "train_lab": train_label},
    )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    logger = setup_logger(args)
    logger.info("Noise ratio: {}".format(args.noisy_ratio))
    logger.info(args)

    to_seed(args.seed)
    print("...Data loading is beginning...")
    print("The noise_ratio is: ", args.noisy_ratio)
    input_data_par = get_loader(
        args.dataset,
        args.batch_size,
        args.noisy_ratio,
        args.noise_mode,
        data_root=args.data_root,
        noise_root=args.noise_root,
    )
    print("...Data loading is completed...")
    args.data_class = input_data_par["num_class"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft = IDCM_NN(
        img_input_dim=input_data_par["img_dim"],
        text_input_dim=input_data_par["text_dim"],
        output_dim=args.output_dim,
        num_class=input_data_par["num_class"],
    ).to(device)
    optimizer = optim.Adam([{"params": model_ft.parameters(), "lr": args.lr}])

    print("...Training is beginning...")
    if args.method == "ssmh":
        model_ft, history = train_model_ssmh(model_ft, input_data_par, optimizer, args)
    else:
        model_ft, history = train_model_synchronous(model_ft, input_data_par, optimizer, args)
    print("...Training is completed...")

    evaluate_and_save_features(model_ft, input_data_par, args, logger, history)
