#!/usr/bin/env python3
"""
fixed_chain_design_example.py

Function：演示Fixed-chainMode的蛋白质sequence设计
Fixed一条chain，设计另一条chain的sequence

Usage：
python fixed_chain_design_example.py
"""

import os
import sys
import torch
import argparse

# 添加项目根directory到Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_complex_mpnn import ProteinMPNNWrapper, set_random_seed, load_config


def design_fixed_chain(model, seq_idx, backbone_coords, fixed_chain_mask):
    """
    使用Fixed-chainMode设计蛋白质sequence
    
    Args:
        model: load的ComplexMPNNModel
        seq_idx: sequence索引张量 (batch_size, seq_len)
        backbone_coords: backboneCoordinates
        fixed_chain_mask: Fixedchain的Mask，True表示Fixed的residue
        
    Returns:
        设计后的sequence
    """
    model.eval()
    
    with torch.no_grad():
        # 前向传播获取logits
        logits = model(seq_idx, backbone_coords, fixed_chain_mask)
        
        # 获取预测的amino acid
        pred_idx = torch.argmax(logits, dim=-1)
    
    return pred_idx


def main():
    """主Function"""
    parser = argparse.ArgumentParser(description='Fixed-chainModesequence设计示例')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best_complexmpnn.pt',
                       help='ModelcheckpointPath')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置filePath')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(42)
    
    # load配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # loadModel
    print("loadModel...")
    model = ProteinMPNNWrapper()
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
        print(f"SuccessloadModel: {args.ckpt}")
    else:
        print(f"Warning: Modelcheckpoint不Exists: {args.ckpt}")
        print("使用Random初始化Model进行演示")
    
    model = model.to(device)
    
    # Create模拟Data（实际使用时应load真实Data）
    print("\nCreate模拟Data...")
    vocab_size = 21
    seq_len = 50
    
    # Randomsequence
    seq_idx = torch.randint(0, vocab_size, (1, seq_len), device=device)
    
    # RandombackboneCoordinates
    backbone_coords = torch.randn(seq_len, 3, 3, device=device)
    
    # Fixed-chainMode：Fixed前半部分，设计后半部分
    fixed_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    fixed_mask[:, seq_len//2:] = False  # 后半部分可设计
    
    print(f"sequence长度: {seq_len}")
    print(f"FixedresidueCount: {fixed_mask.sum().item()}")
    print(f"可设计residueCount: {(~fixed_mask).sum().item()}")
    
    # 进行设计
    print("\nStartFixed-chainMode设计...")
    designed_seq_idx = design_fixed_chain(model, seq_idx, backbone_coords, fixed_mask)
    
    # amino acid映射
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
    
    print(f"原始sequence: {original_seq}")
    print(f"设计sequence: {designed_seq}")
    
    # 标记变化的位置
    changes = []
    for i, (o, d) in enumerate(zip(original_seq, designed_seq)):
        if o != d:
            changes.append(f"{i+1}:{o}→{d}")
    
    print(f"\n变化的residueCount: {len(changes)}")
    if changes:
        print(f"变化详情: {', '.join(changes[:10])}")
        if len(changes) > 10:
            print(f"  ... 还有 {len(changes)-10} 个变化")
    
    print("\n✅ Fixed-chainMode设计Complete！")
    print("\n说明:")
    print("  - 本示例使用模拟Data演示Fixed-chainMode")
    print("  - 实际使用时，请load真实的蛋白质结构Data")
    print("  - Fixed的residue用fixed_mask指定（True表示Fixed）")


if __name__ == "__main__":
    main()
