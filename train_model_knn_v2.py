from __future__ import print_function
from __future__ import division
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time
import copy
from evaluate import fx_calc_map_label, fx_calc_map_multilabel
import numpy as np
from matplotlib import pyplot as plt
from losses import SupConLoss
import scipy.io as sio
import scipy
import scipy.spatial
from sklearn.metrics import accuracy_score
import ot
import random
from scipy.spatial.distance import cdist
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
from load_data import CustomDataSet
def cross_modal_contrastive_ctriterion(fea, args = None):
        n_view = 2
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())
        sim = sim.exp()
        sim = sim - sim.diag().diag()
        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        p1 = diag1 / sim.sum(1)
        loss1 = -(p1).log()

        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        p2 = diag2 / sim.sum(1)
        loss2 = -p2.log()
        return loss1.mean() + loss2.mean()
def rank_loss(features1, features2, margin):
    sim12 = features1.mm(features2.t())
    diag = torch.diag(sim12)
    sim12 = sim12 - diag.view(-1,1) + margin
    sim12[sim12 < 0] = 0

    sim21 = features2.mm(features1.t())
    diag = torch.diag(sim21)
    sim21 = sim21 - diag.view(-1,1) + margin
    sim21[sim21 < 0] = 0
    return sim12.mean() + sim21.mean()

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t()) 
    return Sim


def train_model_synchronous(model, input_data_par, optimizer, args):
    img_train, txt_train, label_train_ori, label_train_noisy = input_data_par['img_train'], input_data_par['text_train'], input_data_par['label_train_ori'], input_data_par['label_train_noisy']
    img_valid, txt_valid, label_valid = input_data_par['img_valid'], input_data_par['text_valid'], input_data_par['label_valid']
    train_clean_indices = np.argmax(label_train_noisy, axis=1) == np.argmax(label_train_ori, axis=1)
    train_clean_indices = np.where(train_clean_indices)[0]
    num_epochs = args.MAX_EPOCH
    since = time.time()
    test_img_acc_history = []
    test_txt_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    MAP_list, Clean_acc_list, Clean_idx_list = [], [], []

    train_dataset = CustomDataSet(img_train, txt_train, label_train_noisy, label_train_ori)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = CustomDataSet(img_valid, txt_valid, label_valid, label_valid)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    # for epoch in range(args.warm_up_epoch):
    #     print('Warm up epoch {}/{}'.format(epoch+1, args.warm_up_epoch))
    #     print('-' * 20)
    #     # 获取质心
    #     barycenters = get_barycenters(model, train_loader, args)
    #     barycenters = torch.tensor(barycenters, requires_grad=False).cuda()
    #     warm_up_epoch(model, train_loader, valid_loader, optimizer, barycenters, args)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)
        # 获取质心
        barycenters = get_barycenters(model, train_loader, args)
        barycenters = torch.tensor(barycenters, requires_grad=False).cuda()
        pure_clean_ids, hard_ids, noisy_ids, img_soft_labels_reordered, txt_soft_labels_reordered = divide_sample(model, train_loader, args)
        img_soft_labels_reordered = img_soft_labels_reordered.detach().cuda()
        txt_soft_labels_reordered = txt_soft_labels_reordered.detach().cuda()

        pure_dataset = CustomDataSet(img_train[pure_clean_ids], txt_train[pure_clean_ids], label_train_noisy[pure_clean_ids], label_train_ori[pure_clean_ids])
        pure_loader = torch.utils.data.DataLoader(pure_dataset, batch_size=args.batch_size, shuffle=True)
        train_pure(model, pure_loader, optimizer, barycenters, args)
        if len(hard_ids) > 1:
            print(len(hard_ids))
            hard_dataset = CustomDataSet(img_train[hard_ids], txt_train[hard_ids], label_train_noisy[hard_ids], label_train_ori[hard_ids])
            hard_loader = torch.utils.data.DataLoader(hard_dataset, batch_size=args.batch_size, shuffle=True)
            train_hard(model, hard_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args)
        if len(noisy_ids) > 1 and args.noisy_train:
            noisy_dataset = CustomDataSet(img_train[noisy_ids], txt_train[noisy_ids], label_train_noisy[noisy_ids], label_train_ori[noisy_ids])
            noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=args.batch_size, shuffle=True)
            train_noisy(model, noisy_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args)
        # 验证
        model.eval()
        t_imgs_fea, t_imgs_pred, t_txts_fea, t_txts_pred, t_labels = [], [], [], [], []
        with torch.no_grad():
            for imgs, txts, labels_noisy, labels_ori, index in valid_loader:
                if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels_ori.cuda()
                t_view1_feature, t_view2_feature = model(imgs, txts)
                t_view1_predict = F.softmax(t_view1_feature.view([t_view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
                t_view2_predict = F.softmax(t_view2_feature.view([t_view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
                t_imgs_fea.append(t_view1_feature.cpu().numpy())
                t_imgs_pred.append(t_view1_predict.cpu().numpy())
                t_txts_fea.append(t_view2_feature.cpu().numpy())
                t_txts_pred.append(t_view2_predict.cpu().numpy())
                t_labels.append(labels.cpu().numpy())
        t_imgs_fea = np.concatenate(t_imgs_fea)
        t_imgs_pred = np.concatenate(t_imgs_pred)
        t_txts_fea = np.concatenate(t_txts_fea)
        t_txts_pred = np.concatenate(t_txts_pred)
                    # t_labels = np.concatenate(t_labels).argmax(1)
        t_labels = np.concatenate(t_labels)
        img2txt = fx_calc_map_multilabel(t_imgs_fea, t_txts_fea, t_labels, metric='cosine')
        txt2img = fx_calc_map_multilabel(t_txts_fea, t_imgs_fea, t_labels, metric='cosine')
        num_val = t_labels.shape[0]
        img_acc = np.sum(np.argmax(t_imgs_pred, axis=1) == np.argmax(t_labels, axis=1)) / num_val
        txt_acc = np.sum(np.argmax(t_txts_pred, axis=1) == np.argmax(t_labels, axis=1)) / num_val

        print('Img2Txt: %.4f  Txt2Img: %.4f Imgacc: %.4f  Txtacc: %.4f lr: %g'%(img2txt, txt2img, img_acc, txt_acc, optimizer.param_groups[0]['lr']))
        if (img2txt + txt2img) / 2 > best_acc:
            best_acc = (img2txt + txt2img) / 2
            best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, MAP_list
def warm_up_epoch(model, train_loader, valid_loader, optimizer, barycenters, args):
    model.train()
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        tmp1 = - (labels * view1_predict.log()).sum(1)
        tmp2 = - (labels * view2_predict.log()).sum(1)
        term1 = tmp1 + tmp2
        # loss = term1.mean()
        # term2 = cross_modal_contrastive_ctriterion([view1_feature, view2_feature])
        term2 = rank_loss(view1_feature, view2_feature, args.margin)
        loss = args.lamda * term1.mean() + args.alpha * term2
        loss.backward()
        optimizer.step()

    model.eval()
    t_imgs_fea, t_imgs_pred, t_txts_fea, t_txts_pred, t_labels = [], [], [], [], []
    with torch.no_grad():
        for imgs, txts, labels_noisy, labels_ori, index in valid_loader:
            if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    txts = txts.cuda()
                    labels = labels_ori.cuda()
            t_view1_feature, t_view2_feature = model(imgs, txts)
            t_view1_predict = F.softmax(t_view1_feature.view([t_view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
            t_view2_predict = F.softmax(t_view2_feature.view([t_view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
            t_imgs_fea.append(t_view1_feature.cpu().numpy())
            t_imgs_pred.append(t_view1_predict.cpu().numpy())
            t_txts_fea.append(t_view2_feature.cpu().numpy())
            t_txts_pred.append(t_view2_predict.cpu().numpy())
            t_labels.append(labels.cpu().numpy())
    t_imgs_fea = np.concatenate(t_imgs_fea)
    t_imgs_pred = np.concatenate(t_imgs_pred)
    t_txts_fea = np.concatenate(t_txts_fea)
    t_txts_pred = np.concatenate(t_txts_pred)
                # t_labels = np.concatenate(t_labels).argmax(1)
    t_labels = np.concatenate(t_labels)
    img2txt = fx_calc_map_multilabel(t_imgs_fea, t_txts_fea, t_labels, metric='cosine')
    txt2img = fx_calc_map_multilabel(t_txts_fea, t_imgs_fea, t_labels, metric='cosine')
    num_val = t_labels.shape[0]
    img_acc = np.sum(np.argmax(t_imgs_pred, axis=1) == np.argmax(t_labels, axis=1)) / num_val
    txt_acc = np.sum(np.argmax(t_txts_pred, axis=1) == np.argmax(t_labels, axis=1)) / num_val

    print('Loss: %.4f Img2Txt: %.4f  Txt2Img: %.4f Imgacc: %.4f  Txtacc: %.4f lr: %g'%(loss, img2txt, txt2img, img_acc, txt_acc, optimizer.param_groups[0]['lr']))
def train_pure(model, train_loader, optimizer, barycenters, args):
    model.train()
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        tmp1 = - (labels * view1_predict.log()).sum(1)
        tmp2 = - (labels * view2_predict.log()).sum(1)
        term1 = tmp1 + tmp2
        # loss = term1.mean()
        # term2 = cross_modal_contrastive_ctriterion([view1_feature, view2_feature])
        term2 = rank_loss(view1_feature, view2_feature, args.margin)
        loss = args.lamda * term1.mean() + args.alpha * term2
        loss.backward()
        optimizer.step()
def train_hard(model, train_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args):
    model.train()
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view1_weight = view1_predict.clone().detach()
        view2_weight = view2_predict.clone().detach()
        weight = 1 - (1 - view1_weight) * (1 - view2_weight)
        if args.hard_weight:
            tmp1 = - (weight * labels * view1_predict.log()).sum(1)
            tmp2 = - (weight * labels * view2_predict.log()).sum(1)
            term1 = tmp1 + tmp2
        else:
            tmp1 = - (labels * view1_predict.log()).sum(1)
            tmp2 = - (labels * view2_predict.log()).sum(1)
            term1 = tmp1 + tmp2
        # loss = term1.mean()
        
        # loss = term1.mean()
        # term2 = cross_modal_contrastive_ctriterion([view1_feature, view2_feature])
        term2 = rank_loss(view1_feature, view2_feature, args.margin)
        loss = args.lamda * term1.mean() + args.alpha * term2
        loss.backward()
        optimizer.step()
def train_noisy(model, train_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args):
    model.train()
    all_pseudo_indices, all_ori_labels = [], []
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        num_classes = view1_predict.shape[1]

        # view1_pesdo_labels = args.lamda * labels + (1 - args.lamda) * img_soft_labels_reordered[index]
        # view2_pesdo_labels = args.lamda * labels + (1 - args.lamda) * txt_soft_labels_reordered[index]
        # pesdo_labels = 1 - (1 - view1_pesdo_labels) * (1 - view2_pesdo_labels)
        # pesdo_labels_idx = torch.argmax(pesdo_labels, dim=1).detach()
        soft_labels = 1 - (1- view1_predict) * (1 - view2_predict)
        pesdo_labels_idx = torch.argmax(soft_labels, dim=1).detach()
        pesdo_labels = F.one_hot(pesdo_labels_idx, num_classes=num_classes).float()
        # 收集伪标签和原始标签
        all_pseudo_indices.extend(pesdo_labels_idx.cpu().numpy())
        all_ori_labels.extend(torch.argmax(labels_ori, dim=1).cpu().numpy())

        tmp1 = (pesdo_labels - view1_predict).abs().sum(1)
        tmp2 = (pesdo_labels - view2_predict).abs().sum(1)
        term1 = tmp1 + tmp2
        # loss = term1.mean()
        # term2 = cross_modal_contrastive_ctriterion([view1_feature, view2_feature])
        term2 = rank_loss(view1_feature, view2_feature, args.margin)
        loss = args.lamda * term1.mean() + args.alpha * term2
        loss.backward()
        optimizer.step()
    # ✅ 计算整个 epoch 的翻新正确率（伪标签 vs 原始噪声标签）
    pseudo_array = np.array(all_pseudo_indices)
    true_array = np.array(all_ori_labels)
    acc = accuracy_score(true_array, pseudo_array)

    print(f"Correct acc: {acc:.4f}")
    

def divide_sample(model, train_loader, args):
    model.eval()
    t_imgs_fea, t_txts_fea, t_labels, t_labels_ori, sample_ids = [], [], [], [], []
    clean_indexs, noisy_indexs = [], []

    with torch.no_grad():
        for imgs, txts, labels_noisy, labels_ori, index in train_loader:
            clean_index = torch.argmax(labels_noisy, dim=1) == torch.argmax(labels_ori, dim=1)
            noisy_index = ~clean_index
            clean_indexs.append(index[clean_index])
            noisy_indexs.append(index[noisy_index])

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels_noisy = labels_noisy.cuda()

            t_view1_feature, t_view2_feature = model(imgs, txts)
            t_imgs_fea.append(t_view1_feature.cpu())
            t_txts_fea.append(t_view2_feature.cpu())
            t_labels.append(labels_noisy.cpu())
            t_labels_ori.append(labels_ori.cpu())
            sample_ids.append(index)

    # 拼接
    t_imgs_fea = torch.cat(t_imgs_fea, dim=0)
    t_txts_fea = torch.cat(t_txts_fea, dim=0)
    t_labels = torch.cat(t_labels, dim=0)
    t_labels_ori = torch.cat(t_labels_ori, dim=0)
    sample_ids = torch.cat(sample_ids, dim=0)
    clean_indexs = torch.cat(clean_indexs, dim=0)
    noisy_indexs = torch.cat(noisy_indexs, dim=0)

    # 将 soft label 处理函数接 numpy 输出
    img_soft_labels = get_soft_labels(t_imgs_fea, t_labels, args.top_k)
    txt_soft_labels = get_soft_labels(t_txts_fea, t_labels, args.top_k)

    # 预测标签（每个模态）
    img_preds = torch.argmax(img_soft_labels, dim=1)
    txt_preds = torch.argmax(txt_soft_labels, dim=1)
    true_labels = torch.argmax(t_labels, dim=1)

    # 判断 clean/hard/noisy（并行）
    img_clean = (img_preds == true_labels)
    txt_clean = (txt_preds == true_labels)

    # 三类掩码
    pure_clean_mask = img_clean & txt_clean
    hard_mask = img_clean ^ txt_clean
    noisy_mask = ~(img_clean | txt_clean)

    # 选择样本 ID
    pure_clean_ids = sample_ids[pure_clean_mask]
    hard_ids = sample_ids[hard_mask]
    noisy_ids = sample_ids[noisy_mask]

    # 精度评估函数
    def compute_selection_accuracy(pred_ids, clean_indexs):
        pred_set = set(pred_ids.cpu().numpy().tolist())
        real_clean_set = set(clean_indexs.cpu().numpy().tolist())
        correct = len(pred_set & real_clean_set)
        accuracy = correct / len(pred_set) if len(pred_set) > 0 else 0.0
        return accuracy, correct

    # 评估 clean/hard 准确率
    clean_acc, clean_num = compute_selection_accuracy(pure_clean_ids, clean_indexs)
    hard_acc, hard_num = compute_selection_accuracy(hard_ids, clean_indexs)
    print(f"Pure-clean samples: Acc:{clean_acc:.4f} Num:{clean_num} Hard-clean samples: Acc: {hard_acc:.4f} Num:{hard_num}")
    # 软标签按照 0,1,2,...,N 重新排列
    img_soft_labels_reordered = torch.zeros_like(img_soft_labels)
    txt_soft_labels_reordered = torch.zeros_like(txt_soft_labels)
    img_soft_labels_reordered[sample_ids] = img_soft_labels
    txt_soft_labels_reordered[sample_ids] = txt_soft_labels

    # 返回值为张量
    return pure_clean_ids, hard_ids, noisy_ids, img_soft_labels_reordered, txt_soft_labels_reordered


def get_barycenters(model, train_loader, args):
    # 计算每个类别的特征质心
    model.eval()
    train_feature_view1, train_feature_view2 = [], []
    train_label = np.array([]).astype('int16')

    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            txts = txts.cuda()
            labels = labels_noisy.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        train_feature_view1.append(view1_feature.cpu().detach().numpy())
        train_feature_view2.append(view2_feature.cpu().detach().numpy())
        train_label = np.concatenate((train_label, np.argmax(labels.cpu().detach().numpy(), axis=1)))
    train_feature_view1 = np.concatenate(train_feature_view1, axis=0)
    train_feature_view2 = np.concatenate(train_feature_view2, axis=0)
    

    barycenters = []
    for class_id in range(labels.shape[1]):
        sample = np.where(train_label==class_id)
        sample_feature_view1 = train_feature_view1[sample]
        sample_feature_view2 = train_feature_view2[sample]
        sample_feature = np.concatenate((sample_feature_view1, sample_feature_view2), axis=0)
        center = k_barycenter(sample_feature.transpose(), args.barycenter_number, args.lambd)
        barycenters += center.transpose().tolist()
    barycenters = np.array(barycenters)
    return barycenters.astype('float32')

def k_barycenter(Q, k, lambd):
    c = len(Q)
    m = len(Q[0])

    H = np.zeros((c,k))
    for i in range(k):
        point = random.randint(0,m-1)
        for j in range(c):
            H[j,i] = Q[j,point]
    
    t = 1.1
    eta = 0.5
    b = np.ones(m)/m

    a1 = np.ones(k)/k
    a2 = np.ones(k)/k
    
    a1_former = np.zeros(k)
    H_former = np.zeros((c,k))
    
    convergence_a1=pow(10,-2)
    convergence_H=pow(10,-2)

    while np.linalg.norm(H-H_former)>convergence_H:

        MHQ = cdist(H.transpose(), Q.transpose(), metric='euclidean')
        a1 = np.ones(k)/k
        a2 = np.ones(k)/k
        a1_former = np.zeros(k)
        while np.linalg.norm(a1-a1_former)>convergence_a1:
            beta = (t+1)/2
            a = (1-1/beta)*a1 + (1/beta)*a2
            result, dual = ot.sinkhorn(a, b, MHQ, lambd, verbose=False, log=True)
            alpha = dual['u']
            alpha = (-beta)*alpha
            alpha = np.exp(alpha)
            
            a2 = a2
            a2_n = a2*alpha
            
            if np.sum(np.isinf(a2_n))==1:
                a2 = np.zeros((len(a2),))
                a2[np.isinf(a2_n)]=1
            elif np.all(a2_n==0):
                a2 = np.ones((len(a2),))/len(a2)
            else:
                a2 = a2_n/np.sum(a2_n)
            
            a1_former = a1
            a1 = (1-1/beta)*a1 + (1/beta)*a2
            t+=1
       
        a = a1
        T = ot.sinkhorn(a, b, MHQ, lambd, verbose=False)
        T = T.transpose()
        diag_a_reverse = np.diag(1/a)
        H_former = H
        H = (1-eta)*H + eta*np.dot(np.dot(Q,T),diag_a_reverse)
    
    return H

def get_soft_labels(features: torch.Tensor, labels: torch.Tensor, top_k: int = 10):
    """
    基于余弦相似度使用 PyTorch 实现的软标签生成函数。
    参数:
        features (Tensor): [N, D] 特征张量，float32
        labels   (Tensor): [N, C] one-hot 标签，float32 或 int
        top_k    (int):    每个样本选取的最近邻个数
    返回:
        soft_labels (Tensor): [N, C] 每个样本的软标签（邻居标签平均）
    """
    
    # 计算余弦相似度矩阵 [N, N]
    sim_matrix = torch.matmul(features, features.T)  # 内积即余弦相似度

    # 排除自身相似度
    N = sim_matrix.shape[0]
    sim_matrix.fill_diagonal_(-1)  # 设置对角线为 -1，排除自身

    # 获取 top_k 个最大相似度的邻居索引
    topk_values, topk_indices = torch.topk(sim_matrix, top_k, dim=1)

    # 利用邻居索引获取邻居标签 [N, top_k, C]
    neighbor_labels = labels[topk_indices]  # 索引是 [N, top_k]，labels 是 [N, C] → 输出 [N, top_k, C]

    # 计算平均标签（作为 soft label）[N, C]
    soft_labels = neighbor_labels.float().mean(dim=1)

    return soft_labels
