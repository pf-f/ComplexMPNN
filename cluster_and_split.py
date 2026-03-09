#!/usr/bin/env python3
"""
cluster_and_split.py

Function：对MPNN数据进行Cluster和数据集Split

Core Rules：
1. 30%sequence一致性Cluster
2. 按clusterSplitTrain/验证/测试集（比例7:2:1）

Dependencies：
- numpy
- scikit-learn

Usage：
python cluster_and_split.py --input_dir data/processed/mpnn_pt --output_dir data/splits
"""

import os
import argparse
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def load_sequences(input_dir):
    """
    load所有MPNN .ptfile中的sequence
    
    Args:
        input_dir: MPNN .ptfiledirectory
    
    Returns:
        字典，键为file名，值为sequence
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
                print(f"load {file} 时出错：{str(e)}")
    
    return sequences


def calculate_sequence_identity(seq1, seq2):
    """
    计算两个sequence之间的一致性
    
    Args:
        seq1: 第一个sequence
        seq2: 第二个sequence
    
    Returns:
        sequence一致性（0-1之间）
    """
    # 计算最短sequence长度
    min_length = min(len(seq1), len(seq2))
    if min_length == 0:
        return 0.0
    
    # 计算匹配的residue数
    matches = sum(1 for a, b in zip(seq1[:min_length], seq2[:min_length]) if a == b)
    
    # 计算一致性
    identity = matches / min_length
    return identity


def build_distance_matrix(sequences):
    """
    Buildsequence之间的距离矩阵
    
    Args:
        sequences: 字典，键为file名，值为sequence
    
    Returns:
        距离矩阵和file名列表
    """
    file_list = list(sequences.keys())
    n = len(file_list)
    
    # 初始化距离矩阵
    distance_matrix = np.zeros((n, n))
    
    # 计算每对sequence之间的距离
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
    对sequence进行Cluster
    
    Args:
        sequences: 字典，键为file名，值为sequence
        identity_threshold: sequence一致性阈值
    
    Returns:
        字典，键为ClusterID，值为该Cluster中的file名列表
    """
    # 如果sequence数量少于2，直接返回每个sequence作为一个Cluster
    if len(sequences) < 2:
        clusters = {}
        for i, file in enumerate(sequences.keys()):
            clusters[i] = [file]
        return clusters
    
    # Build距离矩阵
    distance_matrix, file_list = build_distance_matrix(sequences)
    
    # 使用层次Cluster
    # 距离阈值 = 1 - 一致性阈值
    distance_threshold = 1 - identity_threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average'
    )
    
    # 执行Cluster
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # 整理Cluster结果
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(file_list[i])
    
    return clusters


def split_dataset(clusters, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    按clusterSplit数据集
    
    Args:
        clusters: 字典，键为ClusterID，值为该Cluster中的file名列表
        train_ratio: Train集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        Train集、验证集、测试集file名列表
    """
    # 打乱Cluster顺序
    cluster_list = list(clusters.values())
    np.random.shuffle(cluster_list)
    
    # 计算各集的大小
    total_clusters = len(cluster_list)
    train_size = int(total_clusters * train_ratio)
    val_size = int(total_clusters * val_ratio)
    test_size = total_clusters - train_size - val_size
    
    # 分配Cluster到各集
    train_clusters = cluster_list[:train_size]
    val_clusters = cluster_list[train_size:train_size+val_size]
    test_clusters = cluster_list[train_size+val_size:]
    
    # 展平各集的file名
    train_files = [file for cluster in train_clusters for file in cluster]
    val_files = [file for cluster in val_clusters for file in cluster]
    test_files = [file for cluster in test_clusters for file in cluster]
    
    return train_files, val_files, test_files


def save_split_files(train_files, val_files, test_files, output_dir):
    """
    saveSplit结果到file
    
    Args:
        train_files: Train集file名列表
        val_files: 验证集file名列表
        test_files: 测试集file名列表
        output_dir: outputdirectory
    """
    # 确保outputdirectory存在
    os.makedirs(output_dir, exist_ok=True)
    
    # saveTrain集
    train_file = os.path.join(output_dir, 'train.txt')
    with open(train_file, 'w') as f:
        for file in train_files:
            f.write(f"{file}\n")
    print(f"已saveTrain集到 {train_file}")
    
    # save验证集
    val_file = os.path.join(output_dir, 'val.txt')
    with open(val_file, 'w') as f:
        for file in val_files:
            f.write(f"{file}\n")
    print(f"已saveval到 {val_file}")
    
    # save测试集
    test_file = os.path.join(output_dir, 'test.txt')
    with open(test_file, 'w') as f:
        for file in test_files:
            f.write(f"{file}\n")
    print(f"已savetest到 {test_file}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='对MPNN数据进行Cluster和数据集Split')
    parser.add_argument('--input_dir', required=True, help='MPNN .ptfiledirectory')
    parser.add_argument('--output_dir', default='data/splits', help='outputdirectory')
    parser.add_argument('--identity_threshold', type=float, default=0.3, help='sequence一致性阈值')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='val比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test比例')
    
    args = parser.parse_args()
    
    # loadsequence
    print("loadsequence...")
    sequences = load_sequences(args.input_dir)
    print(f"共load {len(sequences)} 个sequence")
    
    # 如果没有sequence，创建空的Splitfile
    if len(sequences) == 0:
        print("没有sequence可Cluster，创建空的Splitfile...")
        save_split_files([], [], [], args.output_dir)
        print("数据集SplitComplete")
        return
    
    # Cluster
    print("Clustersequence...")
    clusters = cluster_sequences(sequences, args.identity_threshold)
    print(f"共得到 {len(clusters)} 个Cluster")
    
    # Split数据集
    print("Split数据集...")
    train_files, val_files, test_files = split_dataset(
        clusters, args.train_ratio, args.val_ratio, args.test_ratio
    )
    print(f"Train集: {len(train_files)} 个样本")
    print(f"val: {len(val_files)} 个样本")
    print(f"test: {len(test_files)} 个样本")
    
    # saveSplit结果
    print("saveSplit结果...")
    save_split_files(train_files, val_files, test_files, args.output_dir)
    
    print("数据集SplitComplete")


if __name__ == "__main__":
    main()
