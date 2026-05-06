import numpy as np
import pandas as pd



import numpy as np

def min_max_normalization_by_column(data):
    """
    对多维数组按列进行最大最小值归一化处理，输出并删除所有数据值相同的列。

    参数:
    data -- numpy二维数组，其中每列代表一组数据，需要按列归一化。

    返回:
    normalized_data -- 归一化后的numpy二维数组。
    same_values_columns_indices -- 删除的数值相同的列的索引列表。
    """
    # 计算每列的最小值和最大值
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    # 找出所有数据值相同的列（即最大值和最小值相等的列）
    same_values_columns = np.where(max_vals == min_vals)[0]

    # 输出数值相同的列的索引
    if same_values_columns.size > 0:
        print("数值相同的列的索引:", same_values_columns)

    # 如果存在数值相同的列，则删除这些列
    if (max_vals == min_vals).any():
        data = np.delete(data, same_values_columns, axis=1)

    # 重新计算剩下的每列的最小值和最大值
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    # 确保没有全相同的列后进行归一化处理
    normalized_data = (data - min_vals) / (max_vals - min_vals)

    return normalized_data

# 示例使用
# 假设 data 是你的数据集
# data = np.array([...])
# normalized_data, same_values_columns = min_max_normalization_by_column(data)


if __name__ == '__main__':
    data = pd.read_csv('./NIST/X.csv').values
    normalized_data = min_max_normalization_by_column(data)
    print(normalized_data.shape)
    df = pd.DataFrame(normalized_data)
    df.to_csv('./NIST/NIST_normalized.csv', index=False)
