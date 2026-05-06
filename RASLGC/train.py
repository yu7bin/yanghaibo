import os
import random
import torch.cuda as cuda
import torch
from sklearn.cluster import KMeans
import utils
from model import Model,Encoder,ClusterModel
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
from munkres import Munkres
import matplotlib.pyplot as plt
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def train(model: Model, clustermodel: ClusterModel, data: Data, epochs: int = 200,pre_epech:int = 100):
    # 数据准备
    data = data.to(device)
    x, edge_index = data.x, data.edge_index



    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    foptimizer = optim.Adam(list(model.parameters()) + list(clustermodel.parameters()),
                            lr=0.001, weight_decay=1e-5)

    # 初始化记录
    loss_history = []
    acc_history = []
    best_acc = 0
    best_metrics = None
    bestacc = 0





    # 训练循环
    for epoch in range(epochs):
        model.train()
        clustermodel.train()

        # 根据epoch选择优化器
        if epoch < pre_epech:
            current_optimizer = optimizer
        else:
            current_optimizer = foptimizer

        current_optimizer.zero_grad()

        base_z, z1, z2, edge_index2, p1 = model(x, edge_index)

        # 前向传播
        # if epoch < pre_epech:
        #     base_z, z1, z2, edge_index2, p1 = model(x, edge_index)
        # else:
        #     base_z, _, _, _, _ = model(x, edge_index)
        #     q = clustermodel(base_z)
        #     p1 = torch.argmax(q, dim=1).cpu().detach().numpy()
        #     cluster_labels = torch.tensor(p1, device=edge_index.device)
        #     base_z, z1, z2, edge_index2, _ = model(x, edge_index, cluster_labels)

        # 计算损失
        contrastive_loss_val = model.loss(z1, z2)
        recon_loss = model.reconstruct_loss(base_z, edge_index)


        # 预训练获取最优多粒度图结构
        if epoch <= pre_epech:
            # with torch.no_grad():
            #     base_z, _, _, _, _ = model(x, edge_index)
            #     kmeans = KMeans(n_clusters=model.num_clusters, init='k-means++', random_state=0)
            #     p1 = kmeans.fit_predict(base_z.detach().cpu().numpy())
            #     clustermodel.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
            # # 在每个 epoch 结束后计算准确率

            with torch.no_grad():
                z, _, _, _, _ = model(x, edge_index)
                z = F.normalize(z)
                kmeans = KMeans(n_clusters=model.num_clusters, init='k-means++', random_state=0, n_init='auto')
                p1 = kmeans.fit_predict(z.detach().cpu().numpy())
                clustermodel.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)



        if epoch > pre_epech:
            edge_index = edge_index2






        # 仅在后epochs计算KL散度
        if epoch > pre_epech:

            q = clustermodel(base_z)
            # t = model.dual_self_supervised(target_z)
            p = clustermodel.target_distribution(q)
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            # p1 = torch.argmax(q, dim=1).cpu().detach().numpy()
            #计算当轮指标
            z, _, _, _, _ = model(x, edge_index)
            z = F.normalize(z)
            kmeans = KMeans(n_clusters=model.num_clusters, init='k-means++', random_state=0, n_init='auto')
            p1 = kmeans.fit_predict(z.detach().cpu().numpy())
        else:
            kl_loss = 0



        if epoch >= pre_epech:
            # 组合损失
            total_loss = contrastive_loss_val + 0.1 * recon_loss + kl_loss

        else:
            total_loss = contrastive_loss_val + 0.1 * recon_loss


        # 反向传播（确保只进行一次）
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if epoch >= pre_epech:
            torch.nn.utils.clip_grad_norm_(clustermodel.parameters(), 5.0)
        current_optimizer.step()

        # 评估
        with torch.no_grad():
            loss_history.append(total_loss.item())
            y_true = data.y.cpu().numpy()
            y_pred = best_map(y_true, p1)
            acc = metrics.accuracy_score(y_true, y_pred)
            nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
            f1_macro = metrics.f1_score(y_true, y_pred, average='macro')

            print(f"Epoch: {epoch + 1}, Loss: {total_loss.item():.4f}, "
                  f"Loss1: {contrastive_loss_val.item():.4f}, "
                  f"Loss2: {recon_loss.item():.4f}, "
                  f"ACC: {acc:.4f}, NMI: {nmi:.4f}, F1: {f1_macro:.4f}")

            acc_history.append(acc)
            if epoch >= pre_epech:
                if acc > best_acc:
                    best_acc = acc
                    best_metrics = {'ACC': acc, 'NMI': nmi, 'F1': f1_macro}

    # 输出最佳结果
    print("\n最佳结果：")
    print(f"ACC: {best_metrics['ACC']:.4f}, NMI: {best_metrics['NMI']:.4f}, F1: {best_metrics['F1']:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    loss_line = ax1.plot(range(1, epochs + 1), loss_history, color='blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('ACC', color='orange')
    acc_line = ax2.plot(range(1, epochs + 1), acc_history, color='orange', label='ACC')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.legend([loss_line[0], acc_line[0]], ['Loss', 'ACC'], loc='upper left')
    plt.title('Training Loss and Accuracy')
    plt.grid(True)
    plt.show()
    return best_metrics


# 示例使用
# 示例使用
if __name__ == "__main__":
    for i in range(1,4):
        num_runs = 1
        all_metrics = []



        # 打开文件用于写入结果
        with open("dblp_{}results.txt".format(i), "w") as f:
            for run in range(num_runs):
                print(f"运行次数: {run + 1}/{num_runs}")
                cfg = {'seed': 38108}
                torch.manual_seed(cfg['seed'])
                random.seed(cfg['seed'])
                data, num_features = utils.load_dblp(i)
                # 模型参数
                in_dim = num_features
                out_dim = 256
                num_clusters = int(max(data.y)) +1
                begin_idx = int(data.num_nodes * 1)

                # 重新初始化模型
                model = Model(
                    encoder=Encoder(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        activation=nn.PReLU(),
                        k=2
                    ),
                    num_clusters=num_clusters,
                    begin_idx=begin_idx
                )

                clustermodel = ClusterModel(num_clusters, out_dim)

                # 重新设置设备
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                clustermodel = clustermodel.to(device)
                # 开始训练
                metrics_result=train(model, clustermodel, data, epochs=200, pre_epech=100)
                all_metrics.append(metrics_result)

                # 写入每次运行的结果
                f.write(f"第 {run + 1} 次运行结果：\n")
                for key, value in metrics_result.items():
                    f.write(f"  {key}: {value:.4f}\n")

                # 清空 CUDA 缓存（如果使用 GPU）
                if device.type == 'cuda':
                    cuda.empty_cache()

            # 计算平均值和标准差
            avg_metrics = {key: np.mean([metrics[key] for metrics in all_metrics]) for key in all_metrics[0].keys()}
            std_metrics = {key: np.std([metrics[key] for metrics in all_metrics]) for key in all_metrics[0].keys()}

            # 写入平均值
            f.write("十次运行的平均结果：\n")
            for key, value in avg_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")

            # 写入标准差
            f.write("十次运行的标准差：\n")
            for key, value in std_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")

        print("所有结果已保存到 results.txt 文件中。")





