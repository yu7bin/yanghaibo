import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from torch_geometric.utils import dropout_adj
from typing import Dict, List, Tuple
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret



class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.out_channels = out_channels

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_clusters,begin_idx):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
      
        self.fc1 = torch.nn.Linear(encoder.out_channels, encoder.out_channels)
        self.fc2 = torch.nn.Linear(encoder.out_channels, 256)

        # 参数初始化
        self.num_clusters = num_clusters
        self.K = 1  # 最近邻数


        self.begin_idx = begin_idx

    def forward(self, x, edge_index, cluster_labels=None):
        # 第一阶段：生成基础嵌入
        z = self.encoder(x, edge_index)
        base_z = F.normalize(z)

        if cluster_labels is None:
            with torch.no_grad():
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init='auto')
                p1 = kmeans.fit_predict(base_z.detach().cpu().numpy())
                cluster_labels = torch.tensor(p1, device=edge_index.device)

        # 边增强与剪枝
        aug_edge_index = self.add_intra_edges(
            z=base_z,
            labels=cluster_labels,
            edge_index=edge_index
        )
        pruned_edge_index = self.remove_inter_edges(
            labels=cluster_labels,
            edge_index=aug_edge_index
        )

        # 第二阶段：使用动态边生成目标嵌入
        x_drop1 = drop_feature(x, 0.3)
        x_drop2 = drop_feature(x, 0.4)
        edge_index_1 = dropout_adj(edge_index, p=0.2)[0]
        # edge_index_2 = dropout_adj(pruned_edge_index, p=0.0)[0]
        z1 = self.encoder(x_drop1, edge_index_1)
        z2 = self.encoder(x_drop2, pruned_edge_index)

        return z, z1, z2, pruned_edge_index, cluster_labels




    def add_intra_edges(self,
                        z: torch.Tensor,
                        labels: torch.Tensor,
                        edge_index: torch.Tensor) -> torch.Tensor:
        """为begin_idx之前的节点添加同簇边增强，仅计算与同簇内所有节点的相似度"""
        new_edges_list = []

        # 遍历每个簇
        for cluster_id in range(self.num_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = torch.nonzero(cluster_mask, as_tuple=True)[0]

            if cluster_indices.numel() <= self.K:
                continue

            cluster_z = z[cluster_indices]
            cluster_sim_matrix = torch.cdist(cluster_z, cluster_z)

            valid_mask = cluster_indices < self.begin_idx
            valid_indices = cluster_indices[valid_mask]

            if valid_indices.numel() == 0:
                continue

            valid_indices_in_cluster = torch.nonzero(valid_mask, as_tuple=True)[0]

            # 将自身距离设置为无穷大
            valid_sim_matrix = cluster_sim_matrix[valid_indices_in_cluster]
            valid_sim_matrix.fill_diagonal_(float('inf'))

            # 选择最相似的 K 个节点（不包括自身）
            topk_values, topk_indices = torch.topk(
                valid_sim_matrix,
                k=self.K,
                largest=False,
                dim=1
            )

            src = valid_indices.unsqueeze(1).expand(-1, self.K)
            dst = cluster_indices[topk_indices]

            # src = src[:,1:].flatten()  # 展平为一维
            # dst = dst[:,1:].flatten()

            src = src.flatten()  # 展平为一维
            dst = dst.flatten()


            edges = torch.stack([src, dst], dim=0)
            edges = torch.cat([edges, edges.flip(0)], dim=1)  # 无向边
            edges = torch.unique(edges, dim=1)

            if edges.size(1) > 0:
                new_edges_list.append(edges)

        # 合并所有新边
        new_edges = torch.cat(new_edges_list, dim=1) if new_edges_list else \
            torch.empty((2, 0), dtype=torch.long, device=z.device)

        # 正确合并并去重
        combined_edges = torch.cat([edge_index, new_edges], dim=1)  # 保持[2,N]格式
        combined_edges = combined_edges.t().unique(dim=0).t()  # 转置去重后转回

        return combined_edges


    def  remove_inter_edges(self,
                                   labels: torch.Tensor,
                                   edge_index: torch.Tensor) -> torch.Tensor:
        """张量实现的跨簇边剪枝"""
        src, dst = edge_index
        valid_mask = (labels[src] == labels[dst]) | (src > self.begin_idx) & (dst > self.begin_idx)
        return edge_index[:, valid_mask]

    def update_cluster_centers(self, z, z_prime, p1, p2):
        num_clusters = self.num_clusters
        device = z.device

        z= F.normalize(z)
        z_prime= F.normalize(z_prime)

        # 初始化时直接使用正确维度
        u1 = torch.zeros(num_clusters, z.size(1), device=device)
        u2 = torch.zeros(num_clusters, z_prime.size(1), device=device)

        for i in range(num_clusters):
            # 处理原始空间聚类中心
            cluster_mask = (p1 == i)
            cluster_mask = torch.tensor(cluster_mask, device=device)
            cluster_indices = torch.where(cluster_mask)[0]

            if cluster_indices.numel() > 0:
                # 使用更稳健的权重计算方式
                weights = torch.ones_like(cluster_indices, dtype=torch.float) / cluster_indices.numel()

                # 显式维度广播 [m,1]
                weighted_sum = (z[cluster_indices] * weights.view(-1, 1)).sum(dim=0)
                norm = torch.norm(weighted_sum, p=2)

                # 处理零范数情况
                u1[i] = weighted_sum / norm if norm > 1e-8 else weighted_sum

            # 处理增强空间聚类中心
            prime_cluster_mask = (p2 == i)
            prime_cluster_indices = torch.where(prime_cluster_mask)[0]

            if prime_cluster_indices.numel() > 0:
                prime_weights = torch.ones_like(prime_cluster_indices,
                                                dtype=torch.float) / prime_cluster_indices.numel()

                # 显式维度广播 [m,1]
                prime_weighted_sum = (z_prime[prime_cluster_indices] * prime_weights.view(-1, 1)).sum(dim=0)
                prime_norm = torch.norm(prime_weighted_sum, p=2)

                u2[i] = prime_weighted_sum / prime_norm if prime_norm > 1e-8 else prime_weighted_sum

        return u1, u2

    def dual_non_contrastive_clustering_loss(self, u1, u2, tau=0.5):
        num_clusters = self.num_clusters
        loss = 0.0

        for i in range(num_clusters):
            sim = F.cosine_similarity(u1[i].unsqueeze(0), u2, dim=1)
            exp_sim = torch.exp(sim / tau)
            log_prob = torch.log(exp_sim[i] / exp_sim.sum(dim=0))
            loss -= log_prob

        loss /= num_clusters
        return loss


    def reconstruct_loss(self, z, edge_index):
        z=F.normalize(z)
        # mask = (edge_index[0] > self.begin_idx) & (edge_index[1] > self.begin_idx)
        # edge_index = edge_index[:, mask]

        num_edges = edge_index.size(1)
        if num_edges == 0:
            return torch.tensor(0.0, device=z.device)

        src = edge_index[0]
        dst = edge_index[1]

        # 确保索引在有效范围内
        src = src.clamp(max=z.size(0) - 1)
        dst = dst.clamp(max=z.size(0) - 1)

        prob = torch.sigmoid((z[src] * z[dst]).sum(dim=1))
        prob = torch.clamp(prob, min=1e-7, max=1 - 1e-7)  # 避免log(0)
        L_restruct = -torch.sum(torch.log(prob)) / num_edges

        return L_restruct

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.9)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / 0.7)
        indices = torch.arange(0, num_nodes).to(device)
        rand_indices = torch.randperm(num_nodes).to(device)
        losses = []

        #for i in range(num_batches):
        #    mask = indices[i * batch_size:(i + 1) * batch_size]
        #    refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
        #    between_sim = f(self.sim(z1[mask], z2))  # [B, N]

        #    losses.append(-torch.log(
        #        between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
        #        / (refl_sim.sum(1) + between_sim.sum(1)
        #           - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        for i in range(num_batches):
            ordered_mask = indices[i * batch_size:(i + 1) * batch_size]
            random_mask = rand_indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[ordered_mask], z1[random_mask]))  # [B, N]
            between_sim = f(self.sim(z1[ordered_mask], z2[random_mask]))  # [B, N]

            #losses.append(-torch.log(
            #    f((F.normalize(z1[ordered_mask])*F.normalize(z2[ordered_mask])).sum(1))
            #    / (refl_sim.sum(1) + between_sim.sum(1))))
            losses.append(torch.log(refl_sim.sum(1) + between_sim.sum(1)) - (F.normalize(z1[ordered_mask])*F.normalize(z2[ordered_mask])).sum(1)/0.7)

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret





def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


class ClusterModel(torch.nn.Module):
    def __init__(self, n_class, hid_dim, dropout=0,v=1):
        super(ClusterModel, self).__init__()
        # 新增双自监督模块参数
        self.cluster_layer = Parameter(torch.Tensor(n_class, hid_dim))  # 聚类中心
        self.v = v  # t-分布自由度
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(hid_dim)  # 添加批归一化层

        # 参数初始化
        self.init_parameters()

    def init_parameters(self):
        # 聚类中心初始化
        nn.init.xavier_normal_(self.cluster_layer)

    # def dual_self_supervised(self, z):
    #     """计算软分配概率"""
    #     # 计算z到各聚类中心的距离 [N, K]
    #     z = F.normalize(z, p=2, dim=1)  # L2归一化
    #     cluster_centers = F.normalize(self.cluster_layer, p=2, dim=1)  # 聚类中心也归一化
    #
    #     # 使用余弦相似度替代欧氏距离
    #     dist = 1 - torch.mm(z, cluster_centers.t())  # [N, K]
    #
    #     # 学生t分布计算
    #     q = 1.0 / (1.0 + dist / self.v)
    #     q = q.pow((self.v + 1.0) / 2.0)
    #
    #     # 归一化得到概率分布
    #     q = (q.t() / torch.sum(q, 1)).t()
    #     return q

    def dual_self_supervised(self, z):
        """计算软分配概率"""

        z = F.normalize(z, p=2, dim=1)  # L2归一化
        cluster_centers = self.cluster_layer  # 聚类中心也归一化

        # 计算欧氏距离
        # [N, 1, dim] - [1, K, dim] -> [N, K, dim]
        dist = (z.unsqueeze(1) - cluster_centers).pow(2).sum(2)  # [N, K]

        # 学生t分布计算
        q = 1.0 / (1.0 + dist / self.v)
        q = q.pow((self.v + 1.0) / 2.0)

        # 归一化得到概率分布
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        # 示例前向传播流程
        x1 = F.relu(self.bn(x1))  # 激活函数和批归一化
        x1 = F.dropout(x1, p=self.dropout, training=self.training)  # dropout

        # 计算聚类分配概率
        q = self.dual_self_supervised(x1)

        # 返回特征和聚类概率
        return q

    def target_distribution(self, q):
        """计算目标分布"""
        # 计算辅助目标分布
        p = q ** 2 / torch.sum(q, 0)
        p = (p.t() / torch.sum(p, 1)).t()
        return p.detach()  # 切断梯度传播


