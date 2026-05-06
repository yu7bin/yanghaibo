import torch
import numpy as np
import pandas as pd


# 创建邻接矩阵

def create_adj(content_tensor, k=6, device='cuda'):
    sample_num = len(content_tensor)

    # 计算曼哈顿距离并获取前 k 个邻居
    edge_index = []

    # 避免存储整个曼哈顿距离矩阵
    for i in range(sample_num):
        # 计算与第 i 个样本的距离
        diff = content_tensor - content_tensor[i].unsqueeze(0)  # 计算差异，确保维度一致
        manhattan_distance = torch.sum(torch.abs(diff), dim=1)  # 计算曼哈顿距离

        # 获取距离最小的 k 个邻居（包括自己）
        sorted_indices = torch.argsort(manhattan_distance)[:k]  # 排序并取前 k 个近邻

        # 将当前样本与其邻居的边索引添加到结果中
        for neighbor in sorted_indices:
            edge_index.append([i, neighbor.item()])

    # 将边索引转换为 PyTorch 张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)

    return edge_index

# 将 PyTorch 张量保存为文本文件
def save_tensor_to_txt(tensor, filename):
    # 确保张量在 CPU 上
    tensor = tensor.cpu()

    # 将张量转换为 NumPy 数组
    np_array = tensor.numpy()

    # 打开文件以写入
    with open(filename, 'w') as f:
        for row in np_array.T:  # 转置以便每一行存储一个边
            row_str = ' '.join(map(str, row))
            # 写入字符串到文件，并添加换行符
            f.write(row_str + '\n')


# 主程序入口
if __name__ == '__main__':
    device = 'cuda'

    # 读取数据
    content = pd.read_csv('Wiki/data1.csv').values
    content = torch.tensor(content, dtype=torch.float32).to(device)


    for i in range(2,11):
        # 创建邻接矩阵
        edge_index = create_adj(content,i)
        # 保存邻接矩阵到文本文件
        save_tensor_to_txt(edge_index, 'Wiki/{}mhdgraph.txt'.format(i))



