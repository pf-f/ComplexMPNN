#!/usr/bin/env python3
"""
interface_recovery_example.py

功能：演示如何计算和使用序列恢复指标
包括界面残基、非界面残基和整体恢复率

使用方法：
python interface_recovery_example.py
"""

import os
import sys
import torch
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interface_recovery import calculate_sequence_recovery
from train_complex_mpnn import (
    ProteinMPNNWrapper, set_random_seed, load_config,
    ComplexMPNNDataSet, collate_fn
)
from torch.utils.data import DataLoader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='序列恢复指标计算示例')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best_complexmpnn.pt',
                       help='模型checkpoint路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--test_split', type=str, default='test.txt',
                       help='测试集切分文件名')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(42)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据集和数据加载器
    print("\n加载测试数据...")
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
    print(f"测试集大小: {len(test_dataset)}")
    
    # 加载模型
    print("\n加载ComplexMPNN模型...")
    model_complex = ProteinMPNNWrapper()
    if os.path.exists(args.ckpt):
        model_complex.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
        print(f"成功加载模型: {args.ckpt}")
    else:
        print(f"警告: 模型checkpoint不存在: {args.ckpt}")
        print("使用随机初始化模型进行演示")
    
    model_complex = model_complex.to(device)
    
    # 创建基线模型（随机初始化）
    print("\n创建基线模型（随机初始化）...")
    model_baseline = ProteinMPNNWrapper()
    model_baseline = model_baseline.to(device)
    print("基线模型创建完成")
    
    # 计算ComplexMPNN的指标
    print("\n" + "="*60)
    print("计算ComplexMPNN的序列恢复指标")
    print("="*60)
    results_complex = calculate_sequence_recovery(
        model_complex, test_dataloader, device, config,
        use_joint_design=True
    )
    print(f"Interface recovery:    {results_complex['interface_recovery']:.4f}")
    print(f"Non-interface recovery: {results_complex['non_interface_recovery']:.4f}")
    print(f"Overall recovery:      {results_complex['overall_recovery']:.4f}")
    print(f"总残基数:             {results_complex['total_residues']}")
    print(f"界面残基数:           {results_complex['total_interface_residues']}")
    print(f"非界面残基数:         {results_complex['total_non_interface_residues']}")
    
    # 计算基线的指标
    print("\n" + "="*60)
    print("计算基线模型的序列恢复指标")
    print("="*60)
    results_baseline = calculate_sequence_recovery(
        model_baseline, test_dataloader, device, config,
        use_joint_design=True
    )
    print(f"Interface recovery:    {results_baseline['interface_recovery']:.4f}")
    print(f"Non-interface recovery: {results_baseline['non_interface_recovery']:.4f}")
    print(f"Overall recovery:      {results_baseline['overall_recovery']:.4f}")
    
    # 性能对比
    print("\n" + "="*60)
    print("性能对比")
    print("="*60)
    print(f"{'指标':<30} {'ComplexMPNN':<15} {'基线':<15} {'提升':<10}")
    print("-" * 70)
    
    for key in ['interface_recovery', 'non_interface_recovery', 'overall_recovery']:
        complex_val = results_complex[key]
        baseline_val = results_baseline[key]
        improvement = complex_val - baseline_val
        print(f"{key:<30} {complex_val:<15.4f} {baseline_val:<15.4f} {improvement:+.4f}")
    
    print("\n" + "="*60)
    print("指标说明")
    print("="*60)
    print("Interface recovery:")
    print("  - 定义：正确预测的界面残基数量 / 总界面残基数量")
    print("  - 重要性：界面残基对蛋白质-蛋白质相互作用至关重要")
    print("  - ComplexMPNN通过3倍权重专门优化此指标")
    print()
    print("Non-interface recovery:")
    print("  - 定义：正确预测的非界面残基数量 / 总非界面残基数量")
    print("  - 说明：非界面残基权重为1，保持与原始ProteinMPNN一致")
    print()
    print("Overall recovery:")
    print("  - 定义：正确预测的总残基数量 / 所有残基数量")
    print("  - 说明：整体序列恢复率，综合衡量模型性能")
    print()
    
    print("✅ 序列恢复指标计算完成！")
    print("\n说明:")
    print("  - 基线模型使用随机初始化，用于对比")
    print("  - 实际使用时，基线应使用未微调的ProteinMPNN预训练权重")
    print("  - 提升值 = ComplexMPNN指标 - 基线指标")
    print("  - 正的提升值表示ComplexMPNN性能更好")


if __name__ == "__main__":
    main()
