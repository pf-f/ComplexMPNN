#!/usr/bin/env python3
"""
loss_functions.py

Function：实现interface-weighted cross entropyloss函数

Core Rules：
1. interface residuesweights设为3，非interface residuesweights设为1
2. 公式：L = (1/N) * sum_i w_i * CE(ŷ_i, y_i)

Dependencies：
- torch

Usage：
python loss_functions.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterfaceWeightedCrossEntropyLoss(nn.Module):
    """
    Interface-weighted cross entropy loss
    
    公式：L = (1/N) * sum_i w_i * CE(ŷ_i, y_i)
    
    其中：
    - w_i = 3 如果residue在界面上
    - w_i = 1 如果residue不在界面上
    """
    
    def __init__(self, interface_weight=3.0, non_interface_weight=1.0):
        """
        初始化loss函数
        
        Args:
            interface_weight: interface residuesweights
            non_interface_weight: 非interface residuesweights
        """
        super().__init__()
        self.interface_weight = interface_weight
        self.non_interface_weight = non_interface_weight
    
    def forward(self, logits, targets, interface_mask):
        """
        计算loss
        
        Args:
            logits: 模型output的logits，形状 (batch_size, seq_len, vocab_size)
            targets: 目标amino acid索引，形状 (batch_size, seq_len)
            interface_mask: 界面掩码，形状 (batch_size, seq_len)，True表示interface residues
        
        Returns:
            加权交叉熵loss
        """
        # 计算每个residue的交叉熵loss
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none'
        )
        
        # 重塑为 (batch_size, seq_len)
        ce_loss = ce_loss.reshape(targets.shape)
        
        # 计算weights
        weights = torch.ones_like(interface_mask, dtype=torch.float32)
        weights[interface_mask] = self.interface_weight
        weights[~interface_mask] = self.non_interface_weight
        
        # 计算加权loss
        weighted_loss = weights * ce_loss
        
        # 计算平均loss
        avg_loss = torch.mean(weighted_loss)
        
        return avg_loss


def test_interface_weighted_cross_entropy():
    """
    测试InterfaceWeightedCrossEntropyLoss
    """
    print("测试InterfaceWeightedCrossEntropyLoss...")
    
    # 创建loss函数
    loss_fn = InterfaceWeightedCrossEntropyLoss(interface_weight=3.0, non_interface_weight=1.0)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    vocab_size = 20
    
    # 随机logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # 随机目标
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 随机界面掩码
    interface_mask = torch.rand(batch_size, seq_len) > 0.7  # 约30%的interface residues
    
    print(f"input形状:")
    print(f"  logits: {logits.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  interface_mask: {interface_mask.shape}")
    
    # 计算loss
    loss = loss_fn(logits, targets, interface_mask)
    
    print(f"interface residues数量: {interface_mask.sum().item()}")
    print(f"非interface residues数量: {(~interface_mask).sum().item()}")
    print(f"加权交叉熵loss: {loss.item():.4f}")
    
    # 验证weights设置
    weights = torch.ones_like(interface_mask, dtype=torch.float32)
    weights[interface_mask] = 3.0
    weights[~interface_mask] = 1.0
    
    print(f"weights验证:")
    print(f"  interface residuesweights: {weights[interface_mask][:3] if interface_mask.sum() > 0 else '无interface residues'}")
    print(f"  非interface residuesweights: {weights[~interface_mask][:3] if (~interface_mask).sum() > 0 else '无非interface residues'}")
    
    print("测试通过！")
    return True


if __name__ == "__main__":
    test_interface_weighted_cross_entropy()
