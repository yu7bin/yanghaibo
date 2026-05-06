import pandas as pd
import numpy as np

# 读取CSV文件
csv_file_path = 'NIST/NIST_normalized.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(csv_file_path)
# 获取行数
n = len(data)

# 生成单位矩阵
identity_matrix = np.eye(n)

# 提取对角线元素及其索引
edges = []
for i in range(n):
    if identity_matrix[i, i] == 1:
        edges.append(f"{i} {i}")  # 将索引格式化为字符串

# 保存为TXT文件
output_txt_path = 'NIST/graph0.txt'  # 指定输出TXT文件的路径
with open(output_txt_path, 'w') as file:
    for edge in edges:
        file.write(edge + '\n')  # 写入每行数据并添加换行符

print(f"单位矩阵的边已保存为 {output_txt_path}")