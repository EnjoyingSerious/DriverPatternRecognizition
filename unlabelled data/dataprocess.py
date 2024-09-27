import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# 1. 读取CSV文件并删除第一行和第一列
def load_data(file_path):
    # 读取csv，跳过第一行（header），并删除第一列（ID列）
    df = pd.read_csv(file_path, skiprows=1)  # 跳过第一行
    df = df.drop(df.columns[0], axis=1)  # 删除第一列（ID列）
    df = df.drop(df.columns[0], axis=1)  # 删除时间戳列

    # 将所有数据转换为浮点数，pandas会自动处理科学记数法
    data = df.astype(float).values
    return data

# 2. 将数据转换为 [N', T, 14] 的形式
def switch_data(data, T):
    N, D = data.shape  # N是总样本数，D是特征数（14）
    N_prime = N // T  # 将N划分为 N' 组，每组 T 个时间步
    data = data[:N_prime * T].reshape(N_prime, T, D)  # 将数据整形成 [N', T, D] 维
    
    return data  