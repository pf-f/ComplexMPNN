#!/usr/bin/env python3
"""
check_full_pipeline.py

Function: Verify that all output files are complete and meet requirements after one-click run

Usage:
python check_full_pipeline.py
"""

import os
import sys
import torch
import pandas as pd


def check_preprocessing():
    """Check if preprocessing data is complete"""
    print("="*60)
    print("Checking Preprocessing Data")
    print("="*60)
    
    checks = []
    
    mpnn_pt_dir = "data/processed/mpnn_pt"
    if os.path.exists(mpnn_pt_dir):
        pt_files = [f for f in os.listdir(mpnn_pt_dir) if f.endswith('.pt')]
        if len(pt_files) >= 10:
            checks.append(("MPNN .pt files", "✅", f"{len(pt_files)} files"))
        else:
            checks.append(("MPNN .pt files", "⚠️", f"Only {len(pt_files)} files (expected >=10)"))
    else:
        checks.append(("MPNN .pt files", "❌", "Directory does not exist"))
    
    interface_masks_dir = "data/processed/interface_masks"
    if os.path.exists(interface_masks_dir):
        mask_files = [f for f in os.listdir(interface_masks_dir) if f.endswith('.pt')]
        if len(mask_files) >= 10:
            checks.append(("Interface mask files", "✅", f"{len(mask_files)} files"))
        else:
            checks.append(("Interface mask files", "⚠️", f"Only {len(mask_files)} files (expected >=10)"))
    else:
        checks.append(("Interface mask files", "❌", "Directory does not exist"))
    
    split_dir = "data/splits"
    if os.path.exists(split_dir):
        for split_file in ['train.txt', 'val.txt', 'test.txt']:
            split_path = os.path.join(split_dir, split_file)
            if os.path.exists(split_path):
                with open(split_path, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                checks.append((f"Dataset split {split_file}", "✅", f"{len(lines)} samples"))
            else:
                checks.append((f"Dataset split {split_file}", "❌", "File does not exist"))
    else:
        checks.append(("Dataset splits", "❌", "Directory does not exist"))
    
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    all_passed = all('❌' not in c[1] for c in checks)
    print()
    return all_passed


def check_training():
    """Check if training results are complete"""
    print("="*60)
    print("Checking Training Results")
    print("="*60)
    
    checks = []
    
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        best_model = "checkpoints/best_complexmpnn.pt"
        if os.path.exists(best_model):
            size_mb = os.path.getsize(best_model) / (1024 * 1024)
            checks.append(("best_model checkpoint", "✅", f"{size_mb:.2f} MB"))
            
            try:
                model_state = torch.load(best_model, map_location='cpu', weights_only=False)
                if isinstance(model_state, dict):
                    param_count = sum(v.numel() for v in model_state.values())
                    checks.append(("Model parameters loaded", "✅", f"{param_count:,} parameters"))
            except Exception as e:
                checks.append(("Model parameters loaded", "❌", str(e)))
        else:
            checks.append(("best_model checkpoint", "❌", "File does not exist"))
        
        epoch_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('complexmpnn_epoch_') and f.endswith('.pt')]
        if epoch_files:
            checks.append(("Regular checkpoints", "✅", f"{len(epoch_files)} epoch files"))
        else:
            checks.append(("Regular checkpoints", "⚠️", "No epoch files found"))
    else:
        checks.append(("Checkpoint directory", "❌", "Directory does not exist"))
    
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    all_passed = all('❌' not in c[1] for c in checks)
    print()
    return all_passed


def check_evaluation():
    """Check if evaluation results are complete"""
    print("="*60)
    print("Checking Evaluation Results")
    print("="*60)
    
    checks = []
    
    eval_dir = "logs/evaluation"
    if os.path.exists(eval_dir):
        recovery_pt = "logs/evaluation/sequence_recovery_results.pt"
        if os.path.exists(recovery_pt):
            checks.append(("Sequence recovery results", "✅", "File exists"))
            try:
                results = torch.load(recovery_pt, map_location='cpu', weights_only=False)
                if 'complex_mpnn' in results and 'baseline' in results:
                    checks.append(("Sequence recovery data", "✅", "Format correct"))
            except Exception as e:
                checks.append(("Sequence recovery data", "❌", str(e)))
        else:
            checks.append(("Sequence recovery results", "❌", "File does not exist"))
        
        csv_files = ['sequence_recovery_results.csv', 'af_multimer_results.csv', 'combined_evaluation_results.csv']
        for csv_file in csv_files:
            csv_path = os.path.join(eval_dir, csv_file)
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    checks.append((f"CSV file {csv_file}", "✅", f"{len(df)} rows"))
                except Exception as e:
                    checks.append((f"CSV file {csv_file}", "❌", str(e)))
            else:
                checks.append((f"CSV file {csv_file}", "❌", "File does not exist"))
        
        png_files = ['sequence_recovery_comparison.png', 'af_multimer_metrics.png']
        for png_file in png_files:
            png_path = os.path.join(eval_dir, png_file)
            if os.path.exists(png_path):
                size_kb = os.path.getsize(png_path) / 1024
                checks.append((f"Visualization {png_file}", "✅", f"{size_kb:.1f} KB"))
            else:
                checks.append((f"Visualization {png_file}", "⚠️", "File does not exist"))
    else:
        checks.append(("Evaluation directory", "❌", "Directory does not exist"))
    
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    all_passed = all('❌' not in c[1] for c in checks)
    print()
    return all_passed


def check_logs():
    """Check log files"""
    print("="*60)
    print("Checking Log Files")
    print("="*60)
    
    checks = []
    
    log_files = ['preprocess.log', 'train.log', 'evaluate.log', 'full_pipeline.log', 'evaluation_test.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            size_kb = os.path.getsize(log_file) / 1024
            checks.append((f"Log {log_file}", "✅", f"{size_kb:.1f} KB"))
        else:
            checks.append((f"Log {log_file}", "⚠️", "File does not exist"))
    
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    print()
    return True


def print_summary(prep_ok, train_ok, eval_ok, logs_ok):
    """Print summary"""
    print("="*60)
    print("Full Pipeline Verification Summary")
    print("="*60)
    
    print(f"Preprocessing data:   {'✅ Passed' if prep_ok else '❌ Failed'}")
    print(f"Training results:     {'✅ Passed' if train_ok else '❌ Failed'}")
    print(f"Evaluation results:   {'✅ Passed' if eval_ok else '❌ Failed'}")
    print(f"Log files:            {'✅ Check complete' if logs_ok else '⚠️ Some missing'}")
    
    print()
    if prep_ok and train_ok and eval_ok:
        print("🎉 All checks passed! Full pipeline verification successful!")
        return True
    else:
        print("⚠️ Some checks did not pass, please check the output above")
        return False


def main():
    """Main function"""
    print("ComplexMPNN Full Pipeline Verification")
    print("Verify that all output files are complete and meet requirements after one-click run")
    print()
    
    prep_ok = check_preprocessing()
    train_ok = check_training()
    eval_ok = check_evaluation()
    logs_ok = check_logs()
    
    success = print_summary(prep_ok, train_ok, eval_ok, logs_ok)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
