#!/usr/bin/env python3
"""
cluster_and_split.py

功能：对MPNN数据进行聚类和数据集切分

核心规则：
1. 30%序列一致性聚类
2. 按cluster切分训练/验证/测试集（比例7:2:1）

依赖：
- numpy
- scikit-learn

使用方法：
python cluster_and_split.py --input_dir data/processed/mpnn_pt --output_dir data/splits
"""

import os
import argparse
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def load_sequences(input_dir):
    """
    加载所有MPNN .pt文件中的序列
    
    Args:
        input_dir: MPNN .pt文件目录
    
    Returns:
        字典，键为文件名，值为序列
    """
    import torch
    sequences = {}
    
    for file in os.listdir(input_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(input_dir, file)
            try:
                data = torch.load(file_path)
                if 'sequence' in data:
                    sequences[file] = data['sequence']
            except Exception as e:
                print(f"加载 {file} 时出错：{str(e)}")
    
    return sequences


def calculate_sequence_identity(seq1, seq2):
    """
    计算两个序列之间的一致性
    
    Args:
        seq1: 第一个序列
        seq2: 第二个序列
    
    Returns:
        序列一致性（0-1之间）
    """
    # 计算最短序列长度
    min_length = min(len(seq1), len(seq2))
    if min_length == 0:
        return 0.0
    
    # 计算匹配的残基数
    matches = sum(1 for a, b in zip(seq1[:min_length], seq2[:min_length]) if a == b)
    
    # 计算一致性
    identity = matches / min_length
    return identity


def build_distance_matrix(sequences):
    """
    构建序列之间的距离矩阵
    
    Args:
        sequences: 字典，键为文件名，值为序列
    
    Returns:
        距离矩阵和文件名列表
    """
    file_list = list(sequences.keys())
    n = len(file_list)
    
    # 初始化距离矩阵
    distance_matrix = np.zeros((n, n))
    
    # 计算每对序列之间的距离
    for i in range(n):
        for j in range(i+1, n):
            seq1 = sequences[file_list[i]]
            seq2 = sequences[file_list[j]]
            identity = calculate_sequence_identity(seq1, seq2)
            # 距离 = 1 - 一致性
            distance = 1 - identity
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix, file_list


def cluster_sequences(sequences, identity_threshold=0.3):
    """
    对序列进行聚类
    
    Args:
        sequences: 字典，键为文件名，值为序列
        identity_threshold: 序列一致性阈值
    
    Returns:
        字典，键为聚类ID，值为该聚类中的文件名列表
    """
    # 如果序列数量少于2，直接返回每个序列作为一个聚类
    if len(sequences) < 2:
        clusters = {}
        for i, file in enumerate(sequences.keys()):
            clusters[i] = [file]
        return clusters
    
    # 构建距离矩阵
    distance_matrix, file_list = build_distance_matrix(sequences)
    
    # 使用层次聚类
    # 距离阈值 = 1 - 一致性阈值
    distance_threshold = 1 - identity_threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average'
    )
    
    # 执行聚类
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # 整理聚类结果
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(file_list[i])
    
    return clusters


def split_dataset(clusters, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    按cluster切分数据集
    
    Args:
        clusters: 字典，键为聚类ID，值为该聚类中的文件名列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        训练集、验证集、测试集文件名列表
    """
    # 打乱聚类顺序
    cluster_list = list(clusters.values())
    np.random.shuffle(cluster_list)
    
    # 计算各集的大小
    total_clusters = len(cluster_list)
    train_size = int(total_clusters * train_ratio)
    val_size = int(total_clusters * val_ratio)
    test_size = total_clusters - train_size - val_size
    
    # 分配聚类到各集
    train_clusters = cluster_list[:train_size]
    val_clusters = cluster_list[train_size:train_size+val_size]
    test_clusters = cluster_list[train_size+val_size:]
    
    # 展平各集的文件名
    train_files = [file for cluster in train_clusters for file in cluster]
    val_files = [file for cluster in val_clusters for file in cluster]
    test_files = [file for cluster in test_clusters for file in cluster]
    
    return train_files, val_files, test_files


def save_split_files(train_files, val_files, test_files, output_dir):
    """
    保存切分结果到文件
    
    Args:
        train_files: 训练集文件名列表
        val_files: 验证集文件名列表
        test_files: 测试集文件名列表
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_file = os.path.join(output_dir, 'train.txt')
    with open(train_file, 'w') as f:
        for file in train_files:
            f.write(f"{file}\n")
    print(f"已保存训练集到 {train_file}")
    
    # 保存验证集
    val_file = os.path.join(output_dir, 'val.txt')
    with open(val_file, 'w') as f:
        for file in val_files:
            f.write(f"{file}\n")
    print(f"已保存验证集到 {val_file}")
    
    # 保存测试集
    test_file = os.path.join(output_dir, 'test.txt')
    with open(test_file, 'w') as f:
        for file in test_files:
            f.write(f"{file}\n")
    print(f"已保存测试集到 {test_file}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='对MPNN数据进行聚类和数据集切分')
    parser.add_argument('--input_dir', required=True, help='MPNN .pt文件目录')
    parser.add_argument('--output_dir', default='data/splits', help='输出目录')
    parser.add_argument('--identity_threshold', type=float, default=0.3, help='序列一致性阈值')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    
    args = parser.parse_args()
    
    # 加载序列
    print("加载序列...")
    sequences = load_sequences(args.input_dir)
    print(f"共加载 {len(sequences)} 个序列")
    
    # 如果没有序列，创建空的切分文件
    if len(sequences) == 0:
        print("没有序列可聚类，创建空的切分文件...")
        save_split_files([], [], [], args.output_dir)
        print("数据集切分完成")
        return
    
    # 聚类
    print("聚类序列...")
    clusters = cluster_sequences(sequences, args.identity_threshold)
    print(f"共得到 {len(clusters)} 个聚类")
    
    # 切分数据集
    print("切分数据集...")
    train_files, val_files, test_files = split_dataset(
        clusters, args.train_ratio, args.val_ratio, args.test_ratio
    )
    print(f"训练集: {len(train_files)} 个样本")
    print(f"验证集: {len(val_files)} 个样本")
    print(f"测试集: {len(test_files)} 个样本")
    
    # 保存切分结果
    print("保存切分结果...")
    save_split_files(train_files, val_files, test_files, args.output_dir)
    
    print("数据集切分完成")


if __name__ == "__main__":
    main()
