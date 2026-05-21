import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.io import loadmat 
# load data
data_name = 'wiki'
view_name = ['Img', 'Txt']
view_id = [0,1] # 选择模态，如果只有一个模态则绘画输入分布图
n_view = len(view_id)
makers = ['^', 's', '*', 'p', 'o']
all_data_features, all_data_labels, modality_num_list = [], [], [0]
# for i in range(n_view):
#     tmp = loadmat('C:/Users/91810/Desktop/pythonProject1/features/' + method_name + '/' + data_name + '_' + str(i) + '_train.mat')
#     all_data_features.append(tmp['train_fea'])
#     all_data_labels.append(tmp['train_lab'].flatten())
#     modality_num_list.append(modality_num_list[-1] + all_data_features[-1].shape[0])
for i in view_id:
    if n_view == 1:
        tmp1 = loadmat('features/' + data_name + '_' + str(i) + '_train.mat')
    else:
        tmp1 = loadmat('features/' + data_name + '_' + str(i) + '_train.mat')
    # tmp2 = loadmat('/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/PRT_projects/pythonProject1/features/RSHCMR/' + method_name + '/' + data_name + '_' + str(i) + '_valid.mat')
    # tmp3 = loadmat('/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/PRT_projects/pythonProject1/features/RSHCMR/' + method_name + '/' + data_name + '_' + str(i) + '.mat')
    features = tmp1['train_fea']
    labels = tmp1['train_lab']
    # 多标签
    # labels = np.argmax(labels, axis=1).flatten()-1
    # 单标签
    labels = labels.flatten() - 1
    all_data_features.append(features)
    all_data_labels.append(labels)
    modality_num_list.append(modality_num_list[-1] + all_data_features[-1].shape[0])
if n_view == 5:
    all_co = np.concatenate((all_data_features[0], all_data_features[1], all_data_features[2],all_data_features[3],all_data_features[4]))
    lable_cos = np.concatenate((all_data_labels[0], all_data_labels[1], all_data_labels[2], all_data_labels[3], all_data_labels[4]))
elif n_view == 2:
    all_co = np.concatenate((all_data_features[0], all_data_features[1]))
    lable_cos = np.concatenate((all_data_labels[0], all_data_labels[1]))
else:
    all_co = all_data_features[0]
    lable_cos = all_data_labels[0]
# ts = TSNE(n_components=2, init='pca', random_state=0 ,n_iter=2000, early_exaggeration=5, perplexity=75)
all_co = np.nan_to_num(all_co)
# prototype_vectors = loadmat('Prototype/' + data_name + '_prototype_' + str(0) + '.mat')['prototype']
# all_co = np.vstack((all_co, prototype_vectors))
ts = TSNE(n_components=2, init='random', random_state=0, early_exaggeration=12, perplexity=50,n_iter=500)
# ts = TSNE(n_components=2, init='random', random_state=0, early_exaggeration=12, perplexity=50,n_iter=2000)
t_feat_all = ts.fit_transform(all_co)
color = ['red', 'gold', 'orange', 'yellow', 'green', 'blue', 'violet', 'pink', 'c', 
         'Teal', 'Lime', 'Aqua', 'Olive', 'SaddleBrown', 'LightBlue', 'LightCoral', 'DeepPink', 'Crimson', 'LightSteelBlue',
         'black']
plt.figure(figsize=(8,8))
# cur_id = 0
# for v in view_id:
#     for i in range(modality_num_list[cur_id], modality_num_list[cur_id+1]): 
#         index = int(lable_cos[i])
#         plt.scatter(t_feat_all[i, 0], t_feat_all[i, 1], edgecolor=color[index], c=color[index], alpha=1,
#         s=30, marker=makers[1], linewidth=1)
#     plt.savefig('Img/' + method_name + '_' + data_name + '_' + view_name[v] +'_representation.pdf')
#     plt.clf()
#     cur_id += 1
    # plt.xlim(-20,20)
    # plt.ylim(-20,20)
plt.clf()
# plt.xlim(-20,20)
# plt.ylim(-20,20)
# plt.xlim(-90,100)
# plt.ylim(-80,90)
# 去掉横坐标和纵坐标
plt.xticks([])
plt.yticks([])
cur_id = 0
for v in view_id:
    for i in range(modality_num_list[cur_id], modality_num_list[cur_id+1]): 
        index = int(lable_cos[i])
        plt.scatter(t_feat_all[i, 0], t_feat_all[i, 1], edgecolor=color[index], c='none', alpha=1,
        s=30, marker=makers[1], linewidth=1)
    cur_id += 1
# C = 10 
# M = 2
# for i in range(C):
#     for j in range(M):
#         index = i * C + j
#         tmp = C * M
#         plt.scatter(t_feat_all[-tmp+index, 0], t_feat_all[-tmp+index, 1], edgecolor='purple', c='none', alpha=1,
#     s=160, linewidth=3, marker='*')
if n_view == 1:
    plt.savefig('img/' + data_name + '_' + str(view_id[0]) + '_' + 'ori_common_representation.pdf')
else:
    plt.savefig('img/' + data_name + '_' + 'common_representation.pdf')