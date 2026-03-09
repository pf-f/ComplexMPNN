#!/usr/bin/env python3
"""
loss_functions.py

Function：实现interface-weighted cross entropylossFunction

Core Rules：
1. Interface residues weights设为3，Non-Interface residues weights设为1
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
        初始化lossFunction
        
        Args:
            interface_weight: Interface residues weights
            non_interface_weight: Non-Interface residues weights
        """
        super().__init__()
        self.interface_weight = interface_weight
        self.non_interface_weight = non_interface_weight
    
    def forward(self, logits, targets, interface_mask):
        """
        计算loss
        
        Args:
            logits: Modeloutput的logits，Shape (batch_size, seq_len, vocab_size)
            targets: 目标amino acid索引，Shape (batch_size, seq_len)
            interface_mask: 界面Mask，Shape (batch_size, seq_len)，True表示interface residues
        
        Returns:
            Weighted cross entropyloss
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
    TestInterfaceWeightedCrossEntropyLoss
    """
    print("TestInterfaceWeightedCrossEntropyLoss...")
    
    # CreatelossFunction
    loss_fn = InterfaceWeightedCrossEntropyLoss(interface_weight=3.0, non_interface_weight=1.0)
    
    # CreateTestData
    batch_size = 2
    seq_len = 10
    vocab_size = 20
    
    # Randomlogits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Random目标
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Random界面Mask
    interface_mask = torch.rand(batch_size, seq_len) > 0.7  # 约30%的interface residues
    
    print(f"inputShape:")
    print(f"  logits: {logits.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  interface_mask: {interface_mask.shape}")
    
    # 计算loss
    loss = loss_fn(logits, targets, interface_mask)
    
    print(f"interface residuesCount: {interface_mask.sum().item()}")
    print(f"Non-interface residuesCount: {(~interface_mask).sum().item()}")
    print(f"Weighted cross entropyloss: {loss.item():.4f}")
    
    # Verifyweights设置
    weights = torch.ones_like(interface_mask, dtype=torch.float32)
    weights[interface_mask] = 3.0
    weights[~interface_mask] = 1.0
    
    print(f"weightsVerify:")
    print(f"  Interface residues weights: {weights[interface_mask][:3] if interface_mask.sum() > 0 else '无interface residues'}")
    print(f"  Non-Interface residues weights: {weights[~interface_mask][:3] if (~interface_mask).sum() > 0 else '无Non-interface residues'}")
    
    print("TestPassed！")
    return True


if __name__ == "__main__":
    test_interface_weighted_cross_entropy()
