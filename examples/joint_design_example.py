#!/usr/bin/env python3
"""
joint_design_example.py

功能：演示Joint-design模式的蛋白质序列设计
同时设计所有链的序列

使用方法：
python joint_design_example.py
"""

import os
import sys
import torch
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_complex_mpnn import ProteinMPNNWrapper, set_random_seed, load_config


def design_joint(model, seq_idx, backbone_coords):
    """
    使用Joint-design模式设计蛋白质序列
    
    Args:
        model: 加载的ComplexMPNN模型
        seq_idx: 序列索引张量 (batch_size, seq_len)
        backbone_coords: 主链坐标
        
    Returns:
        设计后的序列
    """
    model.eval()
    
    # Joint-design模式：所有残基都可设计
    fixed_mask = torch.zeros_like(seq_idx, dtype=torch.bool)
    
    with torch.no_grad():
        # 前向传播获取logits
        logits = model(seq_idx, backbone_coords, fixed_mask)
        
        # 获取预测的氨基酸
        pred_idx = torch.argmax(logits, dim=-1)
    
    return pred_idx


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Joint-design模式序列设计示例')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best_complexmpnn.pt',
                       help='模型checkpoint路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(42)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    model = ProteinMPNNWrapper()
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
        print(f"成功加载模型: {args.ckpt}")
    else:
        print(f"警告: 模型checkpoint不存在: {args.ckpt}")
        print("使用随机初始化模型进行演示")
    
    model = model.to(device)
    
    # 创建模拟数据（实际使用时应加载真实数据）
    print("\n创建模拟数据...")
    vocab_size = 21
    seq_len = 50
    
    # 随机序列
    seq_idx = torch.randint(0, vocab_size, (1, seq_len), device=device)
    
    # 随机主链坐标
    backbone_coords = torch.randn(seq_len, 3, 3, device=device)
    
    print(f"序列长度: {seq_len}")
    print(f"可设计残基数量: {seq_len} (所有残基)")
    
    # 进行设计
    print("\n开始Joint-design模式设计...")
    designed_seq_idx = design_joint(model, seq_idx, backbone_coords)
    
    # 氨基酸映射
    idx_to_aa = {
        0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C',
        5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
        10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P',
        15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V',
        20: 'X'
    }
    
    # 显示结果
    print("\n设计结果:")
    original_seq = ''.join([idx_to_aa[idx.item()] for idx in seq_idx[0]])
    designed_seq = ''.join([idx_to_aa[idx.item()] for idx in designed_seq_idx[0]])
    
    print(f"原始序列: {original_seq}")
    print(f"设计序列: {designed_seq}")
    
    # 标记变化的位置
    changes = []
    for i, (o, d) in enumerate(zip(original_seq, designed_seq)):
        if o != d:
            changes.append(f"{i+1}:{o}→{d}")
    
    print(f"\n变化的残基数量: {len(changes)}")
    if changes:
        print(f"变化详情: {', '.join(changes[:10])}")
        if len(changes) > 10:
            print(f"  ... 还有 {len(changes)-10} 个变化")
    
    print("\n✅ Joint-design模式设计完成！")
    print("\n说明:")
    print("  - 本示例使用模拟数据演示Joint-design模式")
    print("  - 实际使用时，请加载真实的蛋白质结构数据")
    print("  - Joint-design模式中所有残基都可以设计")
    print("  - 适用于同时优化蛋白质复合物的所有链")


if __name__ == "__main__":
    main()
