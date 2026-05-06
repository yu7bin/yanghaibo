import numpy as np
import numpy as np
from sklearn import cluster
from sklearn import metrics
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, Coauthor
import math
import pandas as pd
from torch_geometric.data import Data
from utils_usediffadj import load_dblp

def get_dataset(path, name):
    assert name in ['acm', 'acm', 'acm', 'Physics']

    if name == 'Physics':
        return Coauthor(path, name)
    else:
        return Planetoid(path, name)




def simple_gnn_aggregate(X, E, aggregation='mean'):
    """
    简单的GNN聚合函数，不使用权重矩阵。

    参数:
    X -- 特征矩阵，形状为 (n, d)，其中 n 是节点数，d 是特征维度。
    E -- 边索引矩阵，形状为 (2, m)，其中 m 是边数。
    aggregation -- 聚合函数，可以是 'mean', 'sum', 'max' 等。

    返回:
    X_aggr -- 聚合后的特征矩阵，形状为 (n, d)。
    """
    n, d = X.shape
    m = E.shape[1]

    # 初始化聚合后的特征矩阵，形状为 (n, d)
    X_aggr = np.copy(X)

    # 初始化每个节点的度数（与之相连的边数）
    degree = np.zeros(n)

    # 遍历每条边进行聚合
    for i in range(m):
        start_idx = E[0, i]  # 起始节点索引
        end_idx = E[1, i]  # 结束节点索引

        # 提取两个节点的特征向量
        start_feature = X[start_idx]
        end_feature = X[end_idx]

        # 根据聚合策略进行聚合
        if aggregation == 'mean':
            X_aggr[start_idx] += (start_feature + end_feature) / 2
            X_aggr[end_idx] += (start_feature + end_feature) / 2
        elif aggregation == 'sum':
            X_aggr[start_idx] += start_feature + end_feature
            X_aggr[end_idx] += start_feature + end_feature
        elif aggregation == 'max':
            X_aggr[start_idx] = np.maximum(X_aggr[start_idx], np.maximum(start_feature, end_feature))
            X_aggr[end_idx] = np.maximum(X_aggr[end_idx], np.maximum(start_feature, end_feature))

        # 更新节点的度数
        degree[start_idx] += 1
        degree[end_idx] += 1

    # 将聚合结果归一化，避免除以0
    degree[degree == 0] = 1  # 防止除零错误
    X_aggr /= degree[:, None]  # 对每个节点的聚合特征进行归一化

    return X_aggr


if __name__ == '__main__':
    # path = osp.join(osp.expanduser('~'), 'datasets', 'acm')
    # dataset = get_dataset(path, 'acm')
    # data1 = dataset[0]

    data,_ = load_dblp()
    data.to('cpu')

    X_aggr=simple_gnn_aggregate(data.x.numpy(), data.edge_index.numpy())

    d_x = pd.DataFrame(X_aggr)

    d_x.to_csv('dblp/data1.csv', index=False)

