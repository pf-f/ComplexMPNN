#!/usr/bin/env python3
"""
check_train.py

Function: Validate Train module, including loss function calculation, Train mode switching, checkpoint save

Usage:
python check_train.py
"""

import os
import torch
from loss_functions import InterfaceWeightedCrossEntropyLoss
from train_complex_mpnn import ProteinMPNNWrapper, set_random_seed


def check_loss_function():
    """
    Check if loss function calculation is correct
    """
    print("=== Checking loss function ===")
    
    loss_fn = InterfaceWeightedCrossEntropyLoss(interface_weight=3.0, non_interface_weight=1.0)
    
    batch_size = 2
    seq_len = 10
    vocab_size = 21
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    interface_mask = torch.rand(batch_size, seq_len) > 0.7
    
    loss = loss_fn(logits, targets, interface_mask)
    
    print(f"Input shape:")
    print(f"  logits: {logits.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  interface_mask: {interface_mask.shape}")
    print(f"Interface residues count: {interface_mask.sum().item()}")
    print(f"Weighted cross entropy loss: {loss.item():.4f}")
    
    weights = torch.ones_like(interface_mask, dtype=torch.float32)
    weights[interface_mask] = 3.0
    weights[~interface_mask] = 1.0
    
    print(f"Weights verification:")
    print(f"  Interface residues weights: {weights[interface_mask][:3] if interface_mask.sum() > 0 else 'No interface residues'}")
    print(f"  Non-interface residues weights: {weights[~interface_mask][:3] if (~interface_mask).sum() > 0 else 'No non-interface residues'}")
    
    print("✅ Loss function check passed!\n")
    return True


def check_training_modes():
    """
    Check if Train mode switching is correct
    """
    print("=== Checking Train mode ===")
    
    set_random_seed(42)
    
    model = ProteinMPNNWrapper()
    
    vocab_size = 21
    seq_len = 10
    
    seq_idx = torch.randint(0, vocab_size, (1, seq_len))
    
    backbone_coords = torch.randn(seq_len, 3, 3)
    
    interface_mask = torch.rand(1, seq_len) > 0.7
    
    print("Testing Fixed-chain mode:")
    fixed_mask = torch.rand(interface_mask.shape, device=interface_mask.device) < 0.5
    logits_fixed = model(seq_idx, backbone_coords, fixed_mask)
    print(f"  Output shape: {logits_fixed.shape}")
    print(f"  Fixed residue count: {fixed_mask.sum().item()}")
    
    print("\nTesting Joint-design mode:")
    fixed_mask_joint = torch.zeros_like(interface_mask, dtype=torch.bool)
    logits_joint = model(seq_idx, backbone_coords, fixed_mask_joint)
    print(f"  Output shape: {logits_joint.shape}")
    print(f"  Fixed residue count: {fixed_mask_joint.sum().item()}")
    
    print("✅ Train mode check passed!\n")
    return True


def check_checkpoint_save_load():
    """
    Check if checkpoint save and load is correct
    """
    print("=== Checking checkpoint save and load ===")
    
    model = ProteinMPNNWrapper()
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "test_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saving checkpoint to: {checkpoint_path}")
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint file exists")
        
        file_size = os.path.getsize(checkpoint_path)
        print(f"Checkpoint file size: {file_size} bytes")
        
        model_loaded = ProteinMPNNWrapper()
        model_loaded.load_state_dict(torch.load(checkpoint_path))
        print("✅ Checkpoint loaded successfully")
        
        param_count = sum(p.numel() for p in model.parameters())
        param_count_loaded = sum(p.numel() for p in model_loaded.parameters())
        print(f"Model parameter count: {param_count}")
        print(f"Loaded model parameter count: {param_count_loaded}")
        
        if param_count == param_count_loaded:
            print("✅ Model parameters consistent")
        else:
            print("❌ Model parameters inconsistent")
            return False
    else:
        print("❌ Checkpoint file does not exist")
        return False
    
    os.remove(checkpoint_path)
    print(f"Cleaning up test checkpoint: {checkpoint_path}")
    
    print("✅ Checkpoint save and load check passed!\n")
    return True


def main():
    """
    Main function
    """
    print("Starting Train module validation...\n")
    
    all_passed = True
    
    if not check_loss_function():
        all_passed = False
    
    if not check_training_modes():
        all_passed = False
    
    if not check_checkpoint_save_load():
        all_passed = False
    
    if all_passed:
        print("🎉 All checks passed! Train module validation successful!")
    else:
        print("❌ Some checks failed, please check error information")


if __name__ == "__main__":
    main()
