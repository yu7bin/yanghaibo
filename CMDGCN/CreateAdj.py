import torch
import numpy as np
import pandas as pd

def create_adj(content_tensor, n,device='cuda'):
    sample_num = len(content_tensor)
    # 将内容转换为 PyTorch 张量并移动到 GPU
    # 在 GPU 上初始化距离矩阵
    distance_mat = torch.zeros((sample_num, sample_num), dtype=torch.float32).to(device)

    # 计算欧式距离
    for i in range(sample_num):
        for j in range(sample_num):
            sample_i = content_tensor[i].unsqueeze(0)  # 扩展维度以进行广播
            sample_j = content_tensor[j].unsqueeze(0)
            difference = sample_i - sample_j
            square_difference = difference ** 2
            sum_square_difference = torch.sum(square_difference)
            euclidean_distance = torch.sqrt(sum_square_difference)
            distance_mat[i, j] = 1 / euclidean_distance

    # 将距离矩阵转换回 CPU 并使用 NumPy 进行操作
    distance_mat = distance_mat.cpu().numpy()
    sorted_distance_mat = np.argsort(-distance_mat, axis=1)

    # 初始化邻接矩阵
    adj_matrix = np.zeros((sample_num, sample_num), dtype=np.int32)

    # TODO: 确定 K 的值
    k = n
    # 根据 KNN 构建邻接矩阵
    for i in range(sample_num):
        for j in range(k):
            index = sorted_distance_mat[i][j]
            adj_matrix[i, index] = 1

    # 获取邻接矩阵中非零元素的索引
    row_indices, col_indices = np.nonzero(adj_matrix)

    # 将索引矩阵转换为 PyTorch 长整型张量并移动到 GPU
    edge_index = torch.tensor(np.column_stack((row_indices, col_indices)), dtype=torch.long).to(device)

    return edge_index

# 将 PyTorch 张量保存为文本文件
def save_tensor_to_txt(tensor, filename):
    # 确保张量在 CPU 上
    tensor = tensor.cpu()

    # 将张量转换为 NumPy 数组
    np_array = tensor.numpy()

    # 打开文件以写入
    with open(filename, 'w') as f:
        for row in np_array:
            # 将数组的每一行转换为字符串，并使用空格分隔元素
            row_str = ' '.join(map(str, row))
            # 写入字符串到文件，并添加换行符
            f.write(row_str + '\n')





if __name__ == '__main__':
    device = 'cuda'
    content = pd.read_csv('NIST/NIST_normalized.csv').values
    content = torch.tensor(content, dtype=torch.float32).to(device)
    mid_dim = content.size(1) // 4
    content1 = content[:, :mid_dim]  # 提取前一半特征
    content2 = content[:,2 * mid_dim]  # 提取后一半特征
    content3 = content[:,3 * mid_dim]

    # for i in range(2,11):
    #     edge_index = create_adj(content,i)
    #     save_tensor_to_txt(edge_index, 'NIST/{}graph.txt'.format(i))

    edge_index = create_adj(content, 10)
    save_tensor_to_txt(edge_index, 'NIST/graph.txt')
    edge_index1 = create_adj(content1,10)
    save_tensor_to_txt(edge_index1, 'NIST/graph1.txt')

    edge_index2 = create_adj(content2,10)
    save_tensor_to_txt(edge_index2, 'NIST/graph2.txt')

    edge_index3 = create_adj(content3,10)
    save_tensor_to_txt(edge_index3, 'NIST/graph3.txt')




