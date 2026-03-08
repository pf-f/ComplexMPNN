#!/usr/bin/env python3
"""
loss_functions.py

功能：实现interface-weighted cross entropy损失函数

核心规则：
1. 界面残基权重设为3，非界面残基权重设为1
2. 公式：L = (1/N) * sum_i w_i * CE(ŷ_i, y_i)

依赖：
- torch

使用方法：
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
    - w_i = 3 如果残基在界面上
    - w_i = 1 如果残基不在界面上
    """
    
    def __init__(self, interface_weight=3.0, non_interface_weight=1.0):
        """
        初始化损失函数
        
        Args:
            interface_weight: 界面残基权重
            non_interface_weight: 非界面残基权重
        """
        super().__init__()
        self.interface_weight = interface_weight
        self.non_interface_weight = non_interface_weight
    
    def forward(self, logits, targets, interface_mask):
        """
        计算损失
        
        Args:
            logits: 模型输出的logits，形状 (batch_size, seq_len, vocab_size)
            targets: 目标氨基酸索引，形状 (batch_size, seq_len)
            interface_mask: 界面掩码，形状 (batch_size, seq_len)，True表示界面残基
        
        Returns:
            加权交叉熵损失
        """
        # 计算每个残基的交叉熵损失
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none'
        )
        
        # 重塑为 (batch_size, seq_len)
        ce_loss = ce_loss.reshape(targets.shape)
        
        # 计算权重
        weights = torch.ones_like(interface_mask, dtype=torch.float32)
        weights[interface_mask] = self.interface_weight
        weights[~interface_mask] = self.non_interface_weight
        
        # 计算加权损失
        weighted_loss = weights * ce_loss
        
        # 计算平均损失
        avg_loss = torch.mean(weighted_loss)
        
        return avg_loss


def test_interface_weighted_cross_entropy():
    """
    测试InterfaceWeightedCrossEntropyLoss
    """
    print("测试InterfaceWeightedCrossEntropyLoss...")
    
    # 创建损失函数
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
    interface_mask = torch.rand(batch_size, seq_len) > 0.7  # 约30%的界面残基
    
    print(f"输入形状:")
    print(f"  logits: {logits.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  interface_mask: {interface_mask.shape}")
    
    # 计算损失
    loss = loss_fn(logits, targets, interface_mask)
    
    print(f"界面残基数量: {interface_mask.sum().item()}")
    print(f"非界面残基数量: {(~interface_mask).sum().item()}")
    print(f"加权交叉熵损失: {loss.item():.4f}")
    
    # 验证权重设置
    weights = torch.ones_like(interface_mask, dtype=torch.float32)
    weights[interface_mask] = 3.0
    weights[~interface_mask] = 1.0
    
    print(f"权重验证:")
    print(f"  界面残基权重: {weights[interface_mask][:3] if interface_mask.sum() > 0 else '无界面残基'}")
    print(f"  非界面残基权重: {weights[~interface_mask][:3] if (~interface_mask).sum() > 0 else '无非界面残基'}")
    
    print("测试通过！")
    return True


if __name__ == "__main__":
    test_interface_weighted_cross_entropy()
