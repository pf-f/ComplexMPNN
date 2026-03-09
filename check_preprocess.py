#!/usr/bin/env python3
"""
check_preprocess.py

Function: Verify preprocessing results, including interface_mask and dataset splits meet requirements

Dependencies:
- numpy
- torch

Usage:
python check_preprocess.py --mpnn_dir data/processed/mpnn_pt --split_dir data/splits
"""

import os
import argparse
import numpy as np
import torch


def check_interface_masks(mpnn_dir):
    """
    Check if interface_mask meets requirements
    
    Args:
        mpnn_dir: MPNN .pt file directory
    """
    print("Checking interface_mask...")
    
    mpnn_files = []
    for file in os.listdir(mpnn_dir):
        if file.endswith('.pt'):
            mpnn_files.append(os.path.join(mpnn_dir, file))
    
    print(f"Checking {len(mpnn_files)} MPNN .pt files")
    
    for mpnn_file in mpnn_files:
        try:
            data = torch.load(mpnn_file)
            
            if 'interface_mask' not in data:
                print(f"❌ {os.path.basename(mpnn_file)} missing interface_mask field")
                continue
            
            interface_mask = data['interface_mask']
            if not isinstance(interface_mask, torch.Tensor):
                print(f"❌ {os.path.basename(mpnn_file)} interface_mask type error, should be torch.Tensor")
                continue
            
            if interface_mask.dim() != 1:
                print(f"❌ {os.path.basename(mpnn_file)} interface_mask dimension error, should be 1D")
                continue
            
            if 'sequence' in data:
                sequence_length = len(data['sequence'])
                if len(interface_mask) != sequence_length:
                    print(f"❌ {os.path.basename(mpnn_file)} interface_mask length does not match sequence length")
                    continue
            
            if interface_mask.dtype != torch.bool:
                print(f"❌ {os.path.basename(mpnn_file)} interface_mask data type error, should be bool")
                continue
            
            print(f"✅ {os.path.basename(mpnn_file)} interface_mask check passed")
        except Exception as e:
            print(f"❌ Error checking {os.path.basename(mpnn_file)}: {str(e)}")


def check_dataset_splits(split_dir):
    """
    Check if dataset splits meet requirements
    
    Args:
        split_dir: Dataset split file directory
    """
    print("\nChecking dataset splits...")
    
    split_files = ['train.txt', 'val.txt', 'test.txt']
    for split_file in split_files:
        file_path = os.path.join(split_dir, split_file)
        if not os.path.exists(file_path):
            print(f"❌ {split_file} file does not exist")
            return
    
    train_files = []
    val_files = []
    test_files = []
    
    with open(os.path.join(split_dir, 'train.txt'), 'r') as f:
        train_files = [line.strip() for line in f if line.strip()]
    
    with open(os.path.join(split_dir, 'val.txt'), 'r') as f:
        val_files = [line.strip() for line in f if line.strip()]
    
    with open(os.path.join(split_dir, 'test.txt'), 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]
    
    total_files = len(train_files) + len(val_files) + len(test_files)
    print(f"Training set: {len(train_files)} samples")
    print(f"Validation set: {len(val_files)} samples")
    print(f"Test set: {len(test_files)} samples")
    print(f"Total samples: {total_files}")
    
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)
    
    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)
    
    if overlap_train_val:
        print(f"❌ Training and validation sets have overlap: {overlap_train_val}")
    else:
        print("✅ Training and validation sets have no overlap")
    
    if overlap_train_test:
        print(f"❌ Training and test sets have overlap: {overlap_train_test}")
    else:
        print("✅ Training and test sets have no overlap")
    
    if overlap_val_test:
        print(f"❌ Validation and test sets have overlap: {overlap_val_test}")
    else:
        print("✅ Validation and test sets have no overlap")
    
    if total_files > 0:
        train_ratio = len(train_files) / total_files
        val_ratio = len(val_files) / total_files
        test_ratio = len(test_files) / total_files
        
        print(f"Training set ratio: {train_ratio:.2f}")
        print(f"Validation set ratio: {val_ratio:.2f}")
        print(f"Test set ratio: {test_ratio:.2f}")
        
        if abs(train_ratio - 0.7) < 0.1 and abs(val_ratio - 0.2) < 0.1 and abs(test_ratio - 0.1) < 0.1:
            print("✅ Dataset ratio close to 7:2:1")
        else:
            print("⚠️  Dataset ratio has significant deviation from 7:2:1")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Verify preprocessing results')
    parser.add_argument('--mpnn_dir', default='data/processed/mpnn_pt', help='MPNN .pt file directory')
    parser.add_argument('--split_dir', default='data/splits', help='Dataset split file directory')
    
    args = parser.parse_args()
    
    if os.path.exists(args.mpnn_dir):
        check_interface_masks(args.mpnn_dir)
    else:
        print(f"⚠️  {args.mpnn_dir} directory does not exist, skipping interface_mask check")
    
    if os.path.exists(args.split_dir):
        check_dataset_splits(args.split_dir)
    else:
        print(f"⚠️  {args.split_dir} directory does not exist, skipping dataset split check")


if __name__ == "__main__":
    main()
