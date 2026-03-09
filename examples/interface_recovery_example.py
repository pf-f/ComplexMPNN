#!/usr/bin/env python3
"""
interface_recovery_example.py

Function：演示如何计算和使用sequence恢复Metric
包括interface residues、Non-interface residues和Overall recovery

Usage：
python interface_recovery_example.py
"""

import os
import sys
import torch
import argparse

# 添加项目根directory到Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interface_recovery import calculate_sequence_recovery
from train_complex_mpnn import (
    ProteinMPNNWrapper, set_random_seed, load_config,
    ComplexMPNNDataSet, collate_fn
)
from torch.utils.data import DataLoader


def main():
    """主Function"""
    parser = argparse.ArgumentParser(description='sequence恢复Metric计算示例')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best_complexmpnn.pt',
                       help='ModelcheckpointPath')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置filePath')
    parser.add_argument('--test_split', type=str, default='test.txt',
                       help='testSplitfile名')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(42)
    
    # load配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # CreateTestData集和Dataload器
    print("\nloadTestData...")
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
    
    # loadModel
    print("\nloadComplexMPNNModel...")
    model_complex = ProteinMPNNWrapper()
    if os.path.exists(args.ckpt):
        model_complex.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
        print(f"SuccessloadModel: {args.ckpt}")
    else:
        print(f"Warning: Modelcheckpoint不Exists: {args.ckpt}")
        print("使用Random初始化Model进行演示")
    
    model_complex = model_complex.to(device)
    
    # CreateBaselineModel（Random初始化）
    print("\nCreateBaselineModel（Random初始化）...")
    model_baseline = ProteinMPNNWrapper()
    model_baseline = model_baseline.to(device)
    print("BaselineModelCreateComplete")
    
    # 计算ComplexMPNN的Metric
    print("\n" + "="*60)
    print("计算ComplexMPNN的sequence恢复Metric")
    print("="*60)
    results_complex = calculate_sequence_recovery(
        model_complex, test_dataloader, device, config,
        use_joint_design=True
    )
    print(f"Interface recovery:    {results_complex['interface_recovery']:.4f}")
    print(f"Non-interface recovery: {results_complex['non_interface_recovery']:.4f}")
    print(f"Overall recovery:      {results_complex['overall_recovery']:.4f}")
    print(f"总residue数:             {results_complex['total_residues']}")
    print(f"interface residues数:           {results_complex['total_interface_residues']}")
    print(f"Non-interface residues数:         {results_complex['total_non_interface_residues']}")
    
    # 计算Baseline的Metric
    print("\n" + "="*60)
    print("计算BaselineModel的sequence恢复Metric")
    print("="*60)
    results_baseline = calculate_sequence_recovery(
        model_baseline, test_dataloader, device, config,
        use_joint_design=True
    )
    print(f"Interface recovery:    {results_baseline['interface_recovery']:.4f}")
    print(f"Non-interface recovery: {results_baseline['non_interface_recovery']:.4f}")
    print(f"Overall recovery:      {results_baseline['overall_recovery']:.4f}")
    
    # Performance comparison
    print("\n" + "="*60)
    print("Performance comparison")
    print("="*60)
    print(f"{'Metric':<30} {'ComplexMPNN':<15} {'Baseline':<15} {'Improvement':<10}")
    print("-" * 70)
    
    for key in ['interface_recovery', 'non_interface_recovery', 'overall_recovery']:
        complex_val = results_complex[key]
        baseline_val = results_baseline[key]
        improvement = complex_val - baseline_val
        print(f"{key:<30} {complex_val:<15.4f} {baseline_val:<15.4f} {improvement:+.4f}")
    
    print("\n" + "="*60)
    print("Metric说明")
    print("="*60)
    print("Interface recovery:")
    print("  - 定义：正确预测的interface residuesCount / 总interface residuesCount")
    print("  - 重要性：interface residues对蛋白质-蛋白质相互作用至关重要")
    print("  - ComplexMPNNPassed3倍weights专门优化此Metric")
    print()
    print("Non-interface recovery:")
    print("  - 定义：正确预测的Non-interface residuesCount / 总Non-interface residuesCount")
    print("  - 说明：Non-Interface residues weights为1，保持与原始ProteinMPNNConsistent")
    print()
    print("Overall recovery:")
    print("  - 定义：正确预测的总residueCount / AllresidueCount")
    print("  - 说明：整体sequence恢复率，综合衡量Model性能")
    print()
    
    print("✅ sequence恢复Metric计算Complete！")
    print("\n说明:")
    print("  - BaselineModel使用Random初始化，用于对比")
    print("  - 实际使用时，Baseline应使用未fine-tune的ProteinMPNN预Trainweights")
    print("  - Improvement值 = ComplexMPNNMetric - BaselineMetric")
    print("  - 正的Improvement值表示ComplexMPNN性能更好")


if __name__ == "__main__":
    main()
