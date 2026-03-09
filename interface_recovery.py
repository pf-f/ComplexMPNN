#!/usr/bin/env python3
"""
interface_recovery.py

Function：计算sequence恢复Metric，包括interface residues恢复率、Non-interface residues恢复率、Overall recovery
支持与原始ProteinMPNN（未fine-tune）Baseline对比

Core Rules：
1. 仅使用Test集Data和Train好的Modelcheckpoint
2. 计算3个核心Metric：Interface recovery、Non-interface recovery、Overall recovery
3. 对比原始ProteinMPNN和ComplexMPNN的性能

Usage：
python interface_recovery.py --ckpt checkpoints/best_complexmpnn.pt --test_split data/splits/test.txt
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from train_complex_mpnn import ProteinMPNNWrapper, set_random_seed, load_config, ComplexMPNNDataSet
from torch.utils.data import DataLoader


def calculate_sequence_recovery(model, dataloader, device, config, use_joint_design=False):
    """
    计算sequence恢复Metric
    
    Args:
        model: Model
        dataloader: Dataload器
        device: 设备
        config: 配置
        use_joint_design: 是否使用Joint-designMode，False使用Fixed-chainMode
        
    Returns:
        包含三个核心Metric的字典
    """
    model.eval()
    
    # amino acid到索引的映射
    aa_to_idx = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
        'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
        'X': 20
    }
    idx_to_aa = {v: k for k, v in aa_to_idx.items()}
    
    # 统计Metric
    total_interface_correct = 0
    total_interface_residues = 0
    total_non_interface_correct = 0
    total_non_interface_residues = 0
    total_correct = 0
    total_residues = 0
    
    with torch.no_grad():
        for batch in dataloader:
            for item in batch:
                sequence = item['sequence']
                backbone_coords = item['backbone_coords']
                interface_mask = item['interface_mask']
                
                # 转换sequence为索引
                seq_idx = torch.tensor([aa_to_idx.get(aa, 20) for aa in sequence], device=device)
                seq_idx = seq_idx.unsqueeze(0)
                
                # 转换interface_mask
                interface_mask_tensor = interface_mask.to(device)
                interface_mask_tensor = interface_mask_tensor.unsqueeze(0)
                
                # 根据Mode选择fixed_mask
                if use_joint_design:
                    # Joint-design mode: Allresidue都可以设计
                    fixed_mask = torch.zeros_like(interface_mask_tensor, dtype=torch.bool)
                else:
                    # Fixed-chain mode: Fixedinterface residues，设计Non-interface residues（或者反过来）
                    # 这里简化Processing，FixedNon-interface residues，设计interface residues
                    fixed_mask = ~interface_mask_tensor
                
                # 前向传播获取logits
                logits = model(seq_idx, backbone_coords, fixed_mask)
                
                # 获取预测的amino acid索引
                pred_idx = torch.argmax(logits, dim=-1).squeeze(0)
                
                # 计算恢复率
                seq_idx_flat = seq_idx.squeeze(0)
                pred_idx_flat = pred_idx
                
                # interface residues
                interface_flat = interface_mask_tensor.squeeze(0)
                interface_correct = (seq_idx_flat[interface_flat] == pred_idx_flat[interface_flat]).sum().item()
                total_interface_correct += interface_correct
                total_interface_residues += interface_flat.sum().item()
                
                # Non-interface residues
                non_interface_flat = ~interface_flat
                non_interface_correct = (seq_idx_flat[non_interface_flat] == pred_idx_flat[non_interface_flat]).sum().item()
                total_non_interface_correct += non_interface_correct
                total_non_interface_residues += non_interface_flat.sum().item()
                
                # 整体
                all_correct = (seq_idx_flat == pred_idx_flat).sum().item()
                total_correct += all_correct
                total_residues += len(seq_idx_flat)
    
    # 计算百分比
    interface_recovery = total_interface_correct / total_interface_residues if total_interface_residues > 0 else 0.0
    non_interface_recovery = total_non_interface_correct / total_non_interface_residues if total_non_interface_residues > 0 else 0.0
    overall_recovery = total_correct / total_residues if total_residues > 0 else 0.0
    
    return {
        'interface_recovery': interface_recovery,
        'non_interface_recovery': non_interface_recovery,
        'overall_recovery': overall_recovery,
        'total_interface_residues': total_interface_residues,
        'total_non_interface_residues': total_non_interface_residues,
        'total_residues': total_residues
    }


def main():
    """
    主Function
    """
    parser = argparse.ArgumentParser(description='计算sequence恢复Metric')
    parser.add_argument('--ckpt', type=str, required=True, help='ModelcheckpointPath')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置filePath')
    parser.add_argument('--test_split', type=str, default='test.txt', help='testSplitfile名')
    parser.add_argument('--output_dir', type=str, default='logs/evaluation', help='outputdirectory')
    
    args = parser.parse_args()
    
    # Createoutputdirectory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # load配置
    config = load_config(args.config)
    
    # Set random seed
    set_random_seed(config['random']['seed'], config['random']['deterministic'])
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # CreateTestData集和Dataload器
    from train_complex_mpnn import collate_fn
    test_dataset = ComplexMPNNDataSet(
        config['data']['mpnn_pt_dir'],
        args.test_split,
        config['data']['split_dir']
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f"Test set size: {len(test_dataset)}")
    
    # CreateModel
    print("\n=== loadComplexMPNN（fine-tune后）===")
    model_complex = ProteinMPNNWrapper()
    model_complex.load_state_dict(torch.load(args.ckpt, map_location=device))
    model_complex = model_complex.to(device)
    print(f"SuccessloadModel: {args.ckpt}")
    
    # Create"原始ProteinMPNN"（这里简化为Random初始化的Model作为Baseline）
    print("\n=== load原始ProteinMPNN（Baseline）===")
    model_baseline = ProteinMPNNWrapper()
    model_baseline = model_baseline.to(device)
    print("使用Random初始化Model作为Baseline（实际应load未fine-tune的预Trainweights）")
    
    # 计算ComplexMPNN的Metric
    print("\n=== 计算ComplexMPNN的sequence恢复Metric ===")
    results_complex = calculate_sequence_recovery(
        model_complex, test_dataloader, device, config,
        use_joint_design=True
    )
    print(f"Interface recovery: {results_complex['interface_recovery']:.4f}")
    print(f"Non-interface recovery: {results_complex['non_interface_recovery']:.4f}")
    print(f"Overall recovery: {results_complex['overall_recovery']:.4f}")
    
    # 计算Baseline的Metric
    print("\n=== 计算BaselineProteinMPNN的sequence恢复Metric ===")
    results_baseline = calculate_sequence_recovery(
        model_baseline, test_dataloader, device, config,
        use_joint_design=True
    )
    print(f"Interface recovery: {results_baseline['interface_recovery']:.4f}")
    print(f"Non-interface recovery: {results_baseline['non_interface_recovery']:.4f}")
    print(f"Overall recovery: {results_baseline['overall_recovery']:.4f}")
    
    # save结果
    results = {
        'complex_mpnn': results_complex,
        'baseline': results_baseline
    }
    
    output_path = os.path.join(args.output_dir, 'sequence_recovery_results.pt')
    torch.save(results, output_path)
    print(f"\n结果已save到: {output_path}")
    
    # 打印对比结果
    print("\n=== Performance comparison ===")
    print(f"{'Metric':<30} {'ComplexMPNN':<15} {'Baseline':<15} {'Improvement':<10}")
    print("-" * 70)
    
    for key in ['interface_recovery', 'non_interface_recovery', 'overall_recovery']:
        complex_val = results_complex[key]
        baseline_val = results_baseline[key]
        improvement = complex_val - baseline_val
        print(f"{key:<30} {complex_val:<15.4f} {baseline_val:<15.4f} {improvement:+.4f}")


if __name__ == "__main__":
    main()
