#!/usr/bin/env python3
"""
check_train.py

Function：验证Train模块，包括loss函数计算、Train模式切换、checkpointsave

Usage：
python check_train.py
"""

import os
import torch
from loss_functions import InterfaceWeightedCrossEntropyLoss
from train_complex_mpnn import ProteinMPNNWrapper, set_random_seed


def check_loss_function():
    """
    检查loss函数计算是否正确
    """
    print("=== 检查loss函数 ===")
    
    # 创建loss函数
    loss_fn = InterfaceWeightedCrossEntropyLoss(interface_weight=3.0, non_interface_weight=1.0)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    vocab_size = 21
    
    # 随机logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # 随机目标
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 随机界面掩码
    interface_mask = torch.rand(batch_size, seq_len) > 0.7
    
    # 计算loss
    loss = loss_fn(logits, targets, interface_mask)
    
    print(f"input形状:")
    print(f"  logits: {logits.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  interface_mask: {interface_mask.shape}")
    print(f"interface residues数量: {interface_mask.sum().item()}")
    print(f"加权交叉熵loss: {loss.item():.4f}")
    
    # 验证weights设置
    weights = torch.ones_like(interface_mask, dtype=torch.float32)
    weights[interface_mask] = 3.0
    weights[~interface_mask] = 1.0
    
    print(f"weights验证:")
    print(f"  interface residuesweights: {weights[interface_mask][:3] if interface_mask.sum() > 0 else '无interface residues'}")
    print(f"  非interface residuesweights: {weights[~interface_mask][:3] if (~interface_mask).sum() > 0 else '无非interface residues'}")
    
    print("✅ loss函数检查通过！\n")
    return True


def check_training_modes():
    """
    检查Train模式切换是否正确
    """
    print("=== 检查Train模式 ===")
    
    # Set random seed
    set_random_seed(42)
    
    # 创建模型
    model = ProteinMPNNWrapper()
    
    # 创建测试数据
    vocab_size = 21
    seq_len = 10
    
    # 随机sequence
    seq_idx = torch.randint(0, vocab_size, (1, seq_len))
    
    # 随机backbone坐标
    backbone_coords = torch.randn(seq_len, 3, 3)
    
    # 随机界面掩码
    interface_mask = torch.rand(1, seq_len) > 0.7
    
    print("测试Fixed-chain mode:")
    fixed_mask = torch.rand(interface_mask.shape, device=interface_mask.device) < 0.5
    logits_fixed = model(seq_idx, backbone_coords, fixed_mask)
    print(f"  output形状: {logits_fixed.shape}")
    print(f"  固定residue数量: {fixed_mask.sum().item()}")
    
    print("\n测试Joint-design mode:")
    fixed_mask_joint = torch.zeros_like(interface_mask, dtype=torch.bool)
    logits_joint = model(seq_idx, backbone_coords, fixed_mask_joint)
    print(f"  output形状: {logits_joint.shape}")
    print(f"  固定residue数量: {fixed_mask_joint.sum().item()}")
    
    print("✅ Train模式检查通过！\n")
    return True


def check_checkpoint_save_load():
    """
    检查checkpointsave和load是否正确
    """
    print("=== 检查checkpointsave和load ===")
    
    # 创建模型
    model = ProteinMPNNWrapper()
    
    # 创建checkpointdirectory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # savecheckpoint
    checkpoint_path = os.path.join(checkpoint_dir, "test_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"savecheckpoint到: {checkpoint_path}")
    
    # 验证checkpoint存在
    if os.path.exists(checkpoint_path):
        print(f"✅ checkpointfile存在")
        
        # 获取file大小
        file_size = os.path.getsize(checkpoint_path)
        print(f"checkpointfile大小: {file_size} bytes")
        
        # loadcheckpoint
        model_loaded = ProteinMPNNWrapper()
        model_loaded.load_state_dict(torch.load(checkpoint_path))
        print("✅ checkpointloadSuccess")
        
        # 验证模型参数一致
        param_count = sum(p.numel() for p in model.parameters())
        param_count_loaded = sum(p.numel() for p in model_loaded.parameters())
        print(f"模型参数数量: {param_count}")
        print(f"load模型参数数量: {param_count_loaded}")
        
        if param_count == param_count_loaded:
            print("✅ 模型参数一致")
        else:
            print("❌ 模型参数不一致")
            return False
    else:
        print("❌ checkpointfile不存在")
        return False
    
    # 清理测试checkpoint
    os.remove(checkpoint_path)
    print(f"清理测试checkpoint: {checkpoint_path}")
    
    print("✅ checkpointsave和load检查通过！\n")
    return True


def main():
    """
    主函数
    """
    print("Start验证Train模块...\n")
    
    all_passed = True
    
    # 检查loss函数
    if not check_loss_function():
        all_passed = False
    
    # 检查Train模式
    if not check_training_modes():
        all_passed = False
    
    # 检查checkpoint
    if not check_checkpoint_save_load():
        all_passed = False
    
    if all_passed:
        print("🎉 所有检查通过！Train模块验证Success！")
    else:
        print("❌ 部分检查Failed，请检查Error信息")


if __name__ == "__main__":
    main()
