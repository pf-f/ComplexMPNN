#!/usr/bin/env python3
"""
check_preprocess.py

功能：验证预处理结果，包括interface_mask和数据集切分是否符合要求

依赖：
- numpy
- torch

使用方法：
python check_preprocess.py --mpnn_dir data/processed/mpnn_pt --split_dir data/splits
"""

import os
import argparse
import numpy as np
import torch


def check_interface_masks(mpnn_dir):
    """
    检查interface_mask是否符合要求
    
    Args:
        mpnn_dir: MPNN .pt文件目录
    """
    print("检查interface_mask...")
    
    # 获取所有MPNN .pt文件
    mpnn_files = []
    for file in os.listdir(mpnn_dir):
        if file.endswith('.pt'):
            mpnn_files.append(os.path.join(mpnn_dir, file))
    
    print(f"共检查 {len(mpnn_files)} 个MPNN .pt文件")
    
    # 检查每个文件
    for mpnn_file in mpnn_files:
        try:
            # 加载文件
            data = torch.load(mpnn_file)
            
            # 检查是否包含interface_mask字段
            if 'interface_mask' not in data:
                print(f"❌ {os.path.basename(mpnn_file)} 缺少interface_mask字段")
                continue
            
            # 检查interface_mask类型
            interface_mask = data['interface_mask']
            if not isinstance(interface_mask, torch.Tensor):
                print(f"❌ {os.path.basename(mpnn_file)} interface_mask类型错误，应为torch.Tensor")
                continue
            
            # 检查interface_mask形状
            if interface_mask.dim() != 1:
                print(f"❌ {os.path.basename(mpnn_file)} interface_mask维度错误，应为1维")
                continue
            
            # 检查interface_mask长度是否与序列长度匹配
            if 'sequence' in data:
                sequence_length = len(data['sequence'])
                if len(interface_mask) != sequence_length:
                    print(f"❌ {os.path.basename(mpnn_file)} interface_mask长度与序列长度不匹配")
                    continue
            
            # 检查interface_mask数据类型
            if interface_mask.dtype != torch.bool:
                print(f"❌ {os.path.basename(mpnn_file)} interface_mask数据类型错误，应为bool")
                continue
            
            print(f"✅ {os.path.basename(mpnn_file)} interface_mask检查通过")
        except Exception as e:
            print(f"❌ 检查 {os.path.basename(mpnn_file)} 时出错：{str(e)}")


def check_dataset_splits(split_dir):
    """
    检查数据集切分是否符合要求
    
    Args:
        split_dir: 数据集切分文件目录
    """
    print("\n检查数据集切分...")
    
    # 检查是否存在三个切分文件
    split_files = ['train.txt', 'val.txt', 'test.txt']
    for split_file in split_files:
        file_path = os.path.join(split_dir, split_file)
        if not os.path.exists(file_path):
            print(f"❌ {split_file} 文件不存在")
            return
    
    # 读取三个切分文件
    train_files = []
    val_files = []
    test_files = []
    
    with open(os.path.join(split_dir, 'train.txt'), 'r') as f:
        train_files = [line.strip() for line in f if line.strip()]
    
    with open(os.path.join(split_dir, 'val.txt'), 'r') as f:
        val_files = [line.strip() for line in f if line.strip()]
    
    with open(os.path.join(split_dir, 'test.txt'), 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]
    
    # 检查文件数量
    total_files = len(train_files) + len(val_files) + len(test_files)
    print(f"训练集: {len(train_files)} 个样本")
    print(f"验证集: {len(val_files)} 个样本")
    print(f"测试集: {len(test_files)} 个样本")
    print(f"总样本数: {total_files} 个")
    
    # 检查是否有重叠
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)
    
    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)
    
    if overlap_train_val:
        print(f"❌ 训练集和验证集有重叠：{overlap_train_val}")
    else:
        print("✅ 训练集和验证集无重叠")
    
    if overlap_train_test:
        print(f"❌ 训练集和测试集有重叠：{overlap_train_test}")
    else:
        print("✅ 训练集和测试集无重叠")
    
    if overlap_val_test:
        print(f"❌ 验证集和测试集有重叠：{overlap_val_test}")
    else:
        print("✅ 验证集和测试集无重叠")
    
    # 检查比例
    if total_files > 0:
        train_ratio = len(train_files) / total_files
        val_ratio = len(val_files) / total_files
        test_ratio = len(test_files) / total_files
        
        print(f"训练集比例: {train_ratio:.2f}")
        print(f"验证集比例: {val_ratio:.2f}")
        print(f"测试集比例: {test_ratio:.2f}")
        
        # 检查比例是否接近7:2:1
        if abs(train_ratio - 0.7) < 0.1 and abs(val_ratio - 0.2) < 0.1 and abs(test_ratio - 0.1) < 0.1:
            print("✅ 数据集比例接近7:2:1")
        else:
            print("⚠️  数据集比例与7:2:1有较大偏差")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='验证预处理结果')
    parser.add_argument('--mpnn_dir', default='data/processed/mpnn_pt', help='MPNN .pt文件目录')
    parser.add_argument('--split_dir', default='data/splits', help='数据集切分文件目录')
    
    args = parser.parse_args()
    
    # 检查interface_mask
    if os.path.exists(args.mpnn_dir):
        check_interface_masks(args.mpnn_dir)
    else:
        print(f"⚠️  {args.mpnn_dir} 目录不存在，跳过interface_mask检查")
    
    # 检查数据集切分
    if os.path.exists(args.split_dir):
        check_dataset_splits(args.split_dir)
    else:
        print(f"⚠️  {args.split_dir} 目录不存在，跳过数据集切分检查")


if __name__ == "__main__":
    main()
