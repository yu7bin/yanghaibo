import argparse
import matplotlib.pyplot as plt
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn import cluster
import math
import numpy as np
from utils_usediffadj import *
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from deepmodel_usediffadj import Encoder, Model, drop_feature, SelfExpr, ClusterModel
from eval import label_classification
from utils_usediffadj import enhance_sim_matrix, post_proC, err_rate, best_map

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE


import matplotlib.backends.backend_pdf

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE


# ============================ 1.parameters ==========================
# from visulization import plot_loss, plot_tsne
def plot_tsne(z, labels, output_pdf='tsne_result.pdf'):
    # 使用t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(z)

    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap='tab20', s=10)

    # 去掉网格、坐标轴和文字
    plt.grid(False)
    plt.axis('off')
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')

    # 保存为PDF
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_pdf)
    pdf.savefig(plt.gcf(), bbox_inches='tight', pad_inches=0)
    pdf.close()
    plt.close()



from sklearn.decomposition import PCA


printvals = False

def test(x, edge_index, y):
    gracemodel.eval()
    z = gracemodel(x, edge_index)
    label_classification(z, y, ratio=0.1)


def test_spectral(c, y_train, n_class):
    y_train_x, _ = post_proC(c, n_class, 4, 1)
    print("Spectral Clustering Done.. Finding Best Fit..")
    scores = err_rate(y_train.detach().cpu().numpy(), y_train_x)
    return scores


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--pretrain', type=str, default='T')
    parser.add_argument('--path', type=str, default='.')
    args = parser.parse_args()
    return args


def train(x, edge_index, batch_size, cfg):
    drop_edge_rate_1 = cfg['drop_edge_rate_1']
    drop_edge_rate_2 = cfg['drop_edge_rate_2']
    drop_feature_rate_1 = cfg['drop_feature_rate_1']
    drop_feature_rate_2 = cfg['drop_feature_rate_2']
    gracemodel.train()
    graceoptimizer.zero_grad()
    num_graphs = len(edge_index)

    # 初始化存储丢弃后的边缘索引的列表
    edge_indices_1 = []
    edge_indices_2 = []
    # 遍历每个图
    for i in range(num_graphs):
        # 对每个图应用边缘丢弃率 drop_edge_rate_1
        edge_index_i = edge_index[i]  # 选择第 i 个图的边
        dropped_edge_index_1 = dropout_edge(edge_index_i, p=drop_edge_rate_1)
        edge_indices_1.append(dropped_edge_index_1[0])  # 只保存新的 edge_index

        # 对每个图应用边缘丢弃率 drop_edge_rate_2
        dropped_edge_index_2 = dropout_edge(edge_index_i, p=drop_edge_rate_2)
        edge_indices_2.append(dropped_edge_index_2[0])  # 只保存新的 edge_index

    # 将列表转换回张量形式
    edge_index_1 = edge_indices_1
    edge_index_2 = edge_indices_2


    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = gracemodel(x_1, edge_index_1)
    z2 = gracemodel(x_2, edge_index_2)
    loss = gracemodel.loss(z1, z2, batch_size=batch_size)
    loss.backward()
    graceoptimizer.step()
    return loss.item()

def self_expressive_train(x_train, cfg, n_class):
    max_epoch = cfg['se_epochs']
    alpha = cfg['se_loss_reg']
    patience = cfg['patience']
    losses = []
    x1 = x_train
    best_loss = 1e9
    bad_count = 0
    for epoch in range(max_epoch):
        semodel.train()
        seoptimizer.zero_grad()
        c, x2 = semodel(x1)
        se_loss = torch.norm(x1-x2)
        reg_loss = torch.norm(c)
        loss = se_loss + alpha*reg_loss
        loss.backward()
        seoptimizer.step()
        losses.append(loss.item())
        print('se_loss: {:.9f}'.format(se_loss.item()), 'reg_loss: {:.9f}'.format(reg_loss.item()), end=' ')
        print('full_loss: {:.9f}'.format(loss.item()), flush=True)
        if loss.item()<best_loss:
            if torch.cuda.is_available():
                best_c = c.cpu()
            else:
                best_c = c
            bad_count = 0
            best_loss = loss.item()
        else:
            bad_count += 1
            if bad_count == patience:
                break

    C = best_c
    C = C.cpu().detach().numpy()
    L = enhance_sim_matrix(C, n_class, 4, 1)
     # 绘制损失函数图像
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='seLoss')
    # plt.plot(accs, label='acc', linestyle='--')  # 绘制验证损失
    plt.title('seLoss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return L

# def self_expressive_train(X, n_class):
#     # 初始化 W 和 Q
#     W = torch.rand(n,n_class)  # 假设我们想要分解成10个特征
#     W = W.clamp(min=0)  # 确保W中的所有元素都是非负的
#     Q = torch.rand(10, 50)
#     Q = Q.clamp(min=0)  # 确保Q中的所有元素都是非负的
#
#     # 定义损失函数
#     def nmf_loss(X, W, Q):
#         return torch.norm(X - torch.mm(W, Q.t()), 'fro') ** 2
#
#     # 使用优化器
#     optimizer = optim.Adam([W, Q], lr=0.01)
#
#     # 训练过程
#     num_epochs = 1000
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         loss = nmf_loss(X, W, Q)
#         loss.backward()
#         optimizer.step()
#         if epoch % 100 == 0:
#             print(f'Epoch {epoch}, Loss: {loss.item()}')
#
#     print("Optimized W:", W)
#     print("Optimized Q:", Q)


def batch_cluster_train(x, edge_index, from_list, to_list, val_list,
                        cfg, n_class, batch_size, MODEL_PATH):
    max_epoch = cfg['cluster_epochs']
    alpha = cfg['cluster_loss_reg']
    beta = cfg['final_loss_reg']
    patience = cfg['patience']
    drop_edge_rate_1 = cfg['drop_edge_rate_1']
    drop_edge_rate_2 = cfg['drop_edge_rate_2']
    drop_feature_rate_1 = cfg['drop_feature_rate_1']
    drop_feature_rate_2 = cfg['drop_feature_rate_2']
    best_loss = 1e9
    bad_count = 0
    losses = []  # 记录损失值
    accs = []
    for epoch in range(max_epoch+1):
        gracemodel.train()
        clustermodel.train()
        fulloptimizer.zero_grad()

        num_graphs = len(edge_index)

        # 初始化存储丢弃后的边缘索引的列表
        edge_indices_1 = []
        edge_indices_2 = []
        # 遍历每个图
        for i in range(num_graphs):
            # 对每个图应用边缘丢弃率 drop_edge_rate_1
            edge_index_i = edge_index[i]  # 选择第 i 个图的边
            dropped_edge_index_1 = dropout_edge(edge_index_i, p=drop_edge_rate_1)
            edge_indices_1.append(dropped_edge_index_1[0])  # 只保存新的 edge_index

            # 对每个图应用边缘丢弃率 drop_edge_rate_2
            dropped_edge_index_2 = dropout_edge(edge_index_i, p=drop_edge_rate_2)
            edge_indices_2.append(dropped_edge_index_2[0])  # 只保存新的 edge_index

        # 将列表转换回张量形式
        edge_index_1 = edge_indices_1
        edge_index_2 = edge_indices_2

        x_1 = drop_feature(x, drop_feature_rate_1)
        x_2 = drop_feature(x, drop_feature_rate_2)
        z1 = gracemodel(x_1, edge_index_1)
        z2 = gracemodel(x_2, edge_index_2)
        grace_loss = gracemodel.loss(z1, z2, batch_size)

        z_full = clustermodel(gracemodel(x, edge_index))
        z_from = z_full[from_list]
        z_to = z_full[to_list]
        pred_similarity = torch.sum(z_from*z_to, dim=1)

        numer2 = torch.mm(z_full.T, z_full)
        denom2 = torch.norm(numer2)
        identity_mat = torch.eye(n_class)
        if torch.cuda.is_available():
            identity_mat = identity_mat.cuda()
        B = identity_mat/math.sqrt(n_class)
        C = numer2/denom2
        #
        #给loss1单独加上正则化项
        loss1 = F.mse_loss(pred_similarity, val_list)
        loss2 = torch.norm(B-C)
        loss = beta*grace_loss + loss1 + alpha*loss2 + 0.0001*clustermodel.clustering_loss()
        loss.backward()
        fulloptimizer.step()
        # 记录损失值
        losses.append(loss.item())
        y_p = torch.argmax(z_full, dim=1).cpu().detach().numpy()
        y_t = data.y.cpu().numpy()
        y_p = best_map(y_t, y_p)
        acc = metrics.accuracy_score(y_t, y_p)
        accs.append(acc)

        print('Epoch:', epoch, 'full_loss: {:.5f}'.format(loss.item()), 'grace_loss: {:.5f}'.format(grace_loss.item()), end=' ')
        print('loss1: {:.5f}'.format(loss1.item()), 'loss2: {:.5f}'.format(loss2.item()), flush=True)
        if loss.item()<best_loss:
            bad_count = 0
            best_loss = loss.item()
            torch.save(gracemodel.state_dict(), MODEL_PATH+"gracemodel_boosted")
            torch.save(clustermodel.state_dict(), MODEL_PATH+"clustermodel")
        else:
            bad_count += 1
            print("Model not improved for", bad_count, "consecutive epochs..")
            if bad_count == patience:
                print("Early stopping Cluster Train...")
                continue

        if epoch%10 == 0:
            y_pred = torch.argmax(z_full, dim=1).cpu().detach().numpy()
            y_true = data.y.cpu().numpy()
            y_pred = best_map(y_true,y_pred)
            acc = metrics.accuracy_score(y_true, y_pred)
            nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
            f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
            f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
            print("\n\nAc:", acc, "NMI:", nmi, "F1Ma:", f1_macro, "F1Mi:", f1_micro)

        # 绘制损失函数图像
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss', color='blue')

    # 创建第二个y轴
    ax2 = plt.twinx()
    # 绘制验证准确率在第二个y轴上
    ax2.plot(accs, label='Validation Accuracy', linestyle='--', color='red')

    # 添加标题和轴标签
    plt.title('Training Loss and Validation Accuracy')
    plt.xlabel('Epoch')

    # 设置左边y轴的标签
    plt.ylabel('Loss')
    # 设置右边y轴的标签
    ax2.set_ylabel('Accuracy')

    # 添加图例
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 显示图表
    plt.show()

    return best_loss


if __name__ == '__main__':
    results = []
    for q in range(1):
        for i1 in [1]:
            for j1 in [3]:
                if (j1 + (i1*4)>10):
                    continue
                args = parse_arguments()
                if torch.cuda.is_available():
                    torch.cuda.set_device(args.gpu_id)
                MODEL_PATH = args.path+"/Saved_Models/"+args.dataset+"/"
                # MODEL_PATH = args.path + "/Saved_Models/" + "dblp" + "/"
                cfg = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
                # cfg = yaml.load(open(args.config), Loader=SafeLoader)['dblp']

                torch.manual_seed(cfg['seed'])
                random.seed(cfg['seed'])
                activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[cfg['activation']]
                base_model = ({'GCNConv': GCNConv})[cfg['base_model']]



                data, num_features = load_dblp(1,3)




                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                data = data.to(device)

                plot_tsne(data.x.cpu().numpy(), data.y.cpu().numpy())



                n_nodes = data.x.shape[0]
                n_class = max(data.y).item()+1
                batch_size = cfg['batch_size']

                if batch_size == 0:
                    batch_size = n_nodes

                print("========", args.dataset, "========")
                print("Nodes    :", n_nodes)
                print("Features :", num_features)
                print("Classes  :", n_class)
                print("Distribution :", end=' ')
                for i in range(n_class-1):
                    print(torch.sum(data.y==i).item(), end=',')
                print(torch.sum(data.y==(n_class-1)).item(), end='\n')
                print("======================")

                encoder1 = Encoder(num_features, cfg['num_hidden'], activation,
                                  base_model=base_model, k=cfg['num_layers']).to(device)
                encoder2 = Encoder(cfg['num_hidden'], cfg['num_hidden'], activation,
                                   base_model=base_model, k=cfg['num_layers']).to(device)
                encoder3 = Encoder(cfg['num_hidden'], cfg['num_hidden'], activation,
                                    base_model=base_model, k=cfg['num_layers']).to(device)
                encoder4 = Encoder(cfg['num_hidden'], cfg['num_hidden'], activation,
                                    base_model=base_model, k=cfg['num_layers']).to(device)
                encoder5 = Encoder(cfg['num_hidden'], cfg['num_hidden'], activation,
                                    base_model=base_model, k=cfg['num_layers']).to(device)
                gracemodel = Model(encoder1,encoder2,encoder3,encoder4,encoder5,cfg['num_hidden'], cfg['num_proj_hidden'], num_features,cfg['tau']).to(device)
                graceoptimizer = torch.optim.Adam(gracemodel.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
                semodel          = SelfExpr(batch_size).to(device)
                seoptimizer      = torch.optim.Adam(semodel.parameters(), lr=cfg['se_lr'], weight_decay=cfg['weight_decay'])
                clustermodel = ClusterModel(n_class, cfg['num_hidden']).to(device)
                # clustermodel = ClusterModel(cfg['num_hidden'], cfg['num_cl_hidden'], n_class, cfg['dropout']).to(device)
                clusteroptimizer = torch.optim.Adam(clustermodel.parameters(), lr=cfg['se_lr'], weight_decay=cfg['weight_decay'])

                #============== Pre-training Module ================#
                grace_time = 0
                if args.pretrain=='T':
                    print("Pre-training GRACE model to get baseline embedding for Self Expressive Layer")
                    start = t()
                    prev = start
                    for epoch in range(1, cfg['num_epochs'] + 1):
                        loss = train(data.x, data.edge_index, batch_size, cfg)
                        now = t()
                        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                              f'this epoch {now - prev:.4f}, total {now - start:.4f}', flush=True)
                        prev = now
                    grace_time = t()-start
                    print("Saving pre-trained GRACE Model")
                    torch.save(gracemodel.state_dict(), MODEL_PATH+"gracemodel")

                #============== Self-Expressive Layer training Module ================#
                se_time=0
                if args.pretrain=='T' or args.pretrain=='T1':
                    print("Loading pre-trained GRACE model")
                    gracemodel.load_state_dict(torch.load(MODEL_PATH+"gracemodel"))

                    print("=== Supervised Accuracy test for GRACE Embeddings Generated ===")
                    test(data.x, data.edge_index, data.y)
                    gracemodel.eval()
                    z = gracemodel(data.x, data.edge_index)
                    kmeans = KMeans(n_clusters=n_class,random_state=0)
                    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
                    y_pred_last = y_pred
                    clustermodel.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)


                    z = gracemodel(data.x, data.edge_index)
                    z = z.detach().cpu().numpy()
                    y = data.y.cpu()

                    kmeans = KMeans(n_clusters=n_class, random_state=0)
                    kmeans.fit(z)
                    # 获取聚类标签和聚类中心
                    y_pred = kmeans.labels_
                    y_true = data.y.cpu().numpy()
                    y_pred = best_map(y_true, y_pred)
                    acc = metrics.accuracy_score(y_true, y_pred)
                    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
                    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
                    print("Ac:", acc, "NMI:", nmi, "F1:", f1_macro)
                    tsne = TSNE(n_components=2, init='pca', random_state=0)
                    X_2d = tsne.fit_transform(z)

                    # 可视化降维结果
                    plt.figure(figsize=(6, 5))
                    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolor='k', s=40, cmap=plt.cm.get_cmap('Spectral', n_class))
                    plt.colorbar()
                    plt.title('t-SNE visualization of the dblp dataset by GNN pre-trained')
                    plt.show()







                    start_se = t()
                    X = gracemodel(data.x, data.edge_index).detach()
                    if cfg['normalize']:
                        print("Normalizing embeddings before Self Expressive layer training")
                        X = normalize(X.cpu().numpy())
                        X = torch.tensor(X).to(device)
                    from_list = []
                    to_list = []
                    val_list = []
                    for iters in range(cfg['iterations']):
                        train_labels = random.sample(list(range(n_nodes)), batch_size)
                        x_train = X[train_labels]
                        y_train = data.y[train_labels]
                        print("\n\n\nStarting self expressive train iteration:", iters+1)
                        S = self_expressive_train(x_train, cfg, n_class)

                        print("Performance of Spectral Clustering on the Similarity Matrix")
                        scores = test_spectral(S, y_train, n_class)
                        print(" Ac:", scores[0], "NMI:", scores[1], "F1:", scores[2])

                        print("\nRetriving similarity values for point pairs")
                        count = 0
                        #TODO: Optimize this for faster runtime
                        threshold = cfg['threshold']
                        for i in range(batch_size):
                            for j in range(batch_size):
                                if i == j:
                                    continue
                                if S[i,j]>=(1-threshold) or (S[i,j]<=threshold and S[i,j]>=0):
                                    from_list.append(train_labels[i])
                                    to_list.append(train_labels[j])
                                    val_list.append(S[i,j])
                                    count+=1
                        print("Included values for", count, "points out of", batch_size*batch_size)
                    se_time = t()-start_se
                    print("Self Expressive Layer training done.. time:", se_time)
                    np.save(MODEL_PATH+"from_list.npy", from_list)
                    np.save(MODEL_PATH+"to_list.npy", to_list)
                    np.save(MODEL_PATH+"val_list.npy",val_list)

                #============== Final full training Module ================#
                print("\n\n\nStarting final full training module")
                gracemodel.load_state_dict(torch.load(MODEL_PATH+"gracemodel"))
                from_list = np.load(MODEL_PATH+"from_list.npy")
                to_list = np.load(MODEL_PATH+"to_list.npy")
                val_list = np.load(MODEL_PATH+"val_list.npy")
                start_cluster = t()
                fulloptimizer = torch.optim.Adam((list(gracemodel.parameters()) + list(clustermodel.parameters())), lr=cfg['learning_rate2'], weight_decay=cfg['weight_decay'])
                convergence_loss = batch_cluster_train(data.x, data.edge_index, from_list, to_list, torch.FloatTensor(val_list).to(device),
                                    cfg, n_class, batch_size, MODEL_PATH)
                cluster_time = t()-start_cluster
                print("Final model training done.. time:", cluster_time)
                print("Total training time:", cluster_time+se_time+grace_time)

                print("\n=== Final Testing===")
                gracemodel.load_state_dict(torch.load(MODEL_PATH+"gracemodel_boosted"))
                clustermodel.load_state_dict(torch.load(MODEL_PATH+"clustermodel"))
                clustermodel.eval()
                gracemodel.eval()
                z = clustermodel(gracemodel(data.x, data.edge_index))
                z1 = gracemodel(data.x, data.edge_index).detach().cpu().numpy()



                


                if torch.cuda.is_available():
                    y_pred = torch.argmax(z, dim=1).cpu().detach().numpy()
                    y_true = data.y.cpu().numpy()
                else:
                    y_pred = torch.argmax(z, dim=1).detach().numpy()
                    y_true = data.y.numpy()

                y_pred = best_map(y_true, y_pred)
                acc = metrics.accuracy_score(y_true, y_pred)
                nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
                f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
                print("Convergence Loss:", convergence_loss)
                print("Ac:", acc, "NMI:", nmi, "F1:", f1_macro)
                tsne = TSNE(n_components=2, init='pca', random_state=0)
                X_2d = tsne.fit_transform(z1)

                # 可视化降维结果
                plt.figure(figsize=(6, 5))
                plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolor='k', s=40, cmap=plt.cm.get_cmap('Spectral', n_class))
                plt.colorbar()
                plt.title('t-SNE visualization of the amap dataset by model')
                plt.show()









