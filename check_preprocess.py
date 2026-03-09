#!/usr/bin/env python3
"""
check_preprocess.py

Function：Verify预Processing结果，包括interface_mask和Data集Split是否符合要求

Dependencies：
- numpy
- torch

Usage：
python check_preprocess.py --mpnn_dir data/processed/mpnn_pt --split_dir data/splits
"""

import os
import argparse
import numpy as np
import torch


def check_interface_masks(mpnn_dir):
    """
    Checkinterface_mask是否符合要求
    
    Args:
        mpnn_dir: MPNN .ptfiledirectory
    """
    print("Checkinterface_mask...")
    
    # 获取AllMPNN .ptfile
    mpnn_files = []
    for file in os.listdir(mpnn_dir):
        if file.endswith('.pt'):
            mpnn_files.append(os.path.join(mpnn_dir, file))
    
    print(f"共Check {len(mpnn_files)} 个MPNN .ptfile")
    
    # Check每个file
    for mpnn_file in mpnn_files:
        try:
            # loadfile
            data = torch.load(mpnn_file)
            
            # Check是否包含interface_mask字段
            if 'interface_mask' not in data:
                print(f"❌ {os.path.basename(mpnn_file)} 缺少interface_mask字段")
                continue
            
            # Checkinterface_mask类型
            interface_mask = data['interface_mask']
            if not isinstance(interface_mask, torch.Tensor):
                print(f"❌ {os.path.basename(mpnn_file)} interface_mask类型Error，应为torch.Tensor")
                continue
            
            # Checkinterface_maskShape
            if interface_mask.dim() != 1:
                print(f"❌ {os.path.basename(mpnn_file)} interface_mask维度Error，应为1维")
                continue
            
            # Checkinterface_mask长度是否与sequence长度匹配
            if 'sequence' in data:
                sequence_length = len(data['sequence'])
                if len(interface_mask) != sequence_length:
                    print(f"❌ {os.path.basename(mpnn_file)} interface_mask长度与sequence长度不匹配")
                    continue
            
            # Checkinterface_maskData类型
            if interface_mask.dtype != torch.bool:
                print(f"❌ {os.path.basename(mpnn_file)} interface_maskData类型Error，应为bool")
                continue
            
            print(f"✅ {os.path.basename(mpnn_file)} interface_maskCheckPassed")
        except Exception as e:
            print(f"❌ Check {os.path.basename(mpnn_file)} 时出错：{str(e)}")


def check_dataset_splits(split_dir):
    """
    CheckData集Split是否符合要求
    
    Args:
        split_dir: Data集Splitfiledirectory
    """
    print("\nCheckData集Split...")
    
    # Check是否Exists三个Splitfile
    split_files = ['train.txt', 'val.txt', 'test.txt']
    for split_file in split_files:
        file_path = os.path.join(split_dir, split_file)
        if not os.path.exists(file_path):
            print(f"❌ {split_file} file不Exists")
            return
    
    # 读取三个Splitfile
    train_files = []
    val_files = []
    test_files = []
    
    with open(os.path.join(split_dir, 'train.txt'), 'r') as f:
        train_files = [line.strip() for line in f if line.strip()]
    
    with open(os.path.join(split_dir, 'val.txt'), 'r') as f:
        val_files = [line.strip() for line in f if line.strip()]
    
    with open(os.path.join(split_dir, 'test.txt'), 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]
    
    # CheckfileCount
    total_files = len(train_files) + len(val_files) + len(test_files)
    print(f"Train集: {len(train_files)} 个样本")
    print(f"val: {len(val_files)} 个样本")
    print(f"test: {len(test_files)} 个样本")
    print(f"总样本数: {total_files} 个")
    
    # Check是否有重叠
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)
    
    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)
    
    if overlap_train_val:
        print(f"❌ Train集和Verify集有重叠：{overlap_train_val}")
    else:
        print("✅ Train集和Verify集无重叠")
    
    if overlap_train_test:
        print(f"❌ Train集和Test集有重叠：{overlap_train_test}")
    else:
        print("✅ Train集和Test集无重叠")
    
    if overlap_val_test:
        print(f"❌ Verify集和Test集有重叠：{overlap_val_test}")
    else:
        print("✅ Verify集和Test集无重叠")
    
    # Check比例
    if total_files > 0:
        train_ratio = len(train_files) / total_files
        val_ratio = len(val_files) / total_files
        test_ratio = len(test_files) / total_files
        
        print(f"Train集比例: {train_ratio:.2f}")
        print(f"val比例: {val_ratio:.2f}")
        print(f"test比例: {test_ratio:.2f}")
        
        # Check比例是否接近7:2:1
        if abs(train_ratio - 0.7) < 0.1 and abs(val_ratio - 0.2) < 0.1 and abs(test_ratio - 0.1) < 0.1:
            print("✅ Data集比例接近7:2:1")
        else:
            print("⚠️  Data集比例与7:2:1有较大偏差")


def main():
    """
    主Function
    """
    parser = argparse.ArgumentParser(description='Verify预Processing结果')
    parser.add_argument('--mpnn_dir', default='data/processed/mpnn_pt', help='MPNN .ptfiledirectory')
    parser.add_argument('--split_dir', default='data/splits', help='Data集Splitfiledirectory')
    
    args = parser.parse_args()
    
    # Checkinterface_mask
    if os.path.exists(args.mpnn_dir):
        check_interface_masks(args.mpnn_dir)
    else:
        print(f"⚠️  {args.mpnn_dir} directory不Exists，Skipinterface_maskCheck")
    
    # CheckData集Split
    if os.path.exists(args.split_dir):
        check_dataset_splits(args.split_dir)
    else:
        print(f"⚠️  {args.split_dir} directory不Exists，SkipData集SplitCheck")


if __name__ == "__main__":
    main()
