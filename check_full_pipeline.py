#!/usr/bin/env python3
"""
check_full_pipeline.py

Function：Verify一键运行后Alloutputfile是否完整且符合要求

Usage：
python check_full_pipeline.py
"""

import os
import sys
import torch
import pandas as pd


def check_preprocessing():
    """Check预ProcessingData是否完整"""
    print("="*60)
    print("Check预ProcessingData")
    print("="*60)
    
    checks = []
    
    # Checkmpnn_ptdirectory
    mpnn_pt_dir = "data/processed/mpnn_pt"
    if os.path.exists(mpnn_pt_dir):
        pt_files = [f for f in os.listdir(mpnn_pt_dir) if f.endswith('.pt')]
        if len(pt_files) >= 10:
            checks.append(("MPNN .ptfile", "✅", f"{len(pt_files)}个file"))
        else:
            checks.append(("MPNN .ptfile", "⚠️", f"只有{len(pt_files)}个file（期望>=10）"))
    else:
        checks.append(("MPNN .ptfile", "❌", "directory不Exists"))
    
    # Checkinterface_masksdirectory
    interface_masks_dir = "data/processed/interface_masks"
    if os.path.exists(interface_masks_dir):
        mask_files = [f for f in os.listdir(interface_masks_dir) if f.endswith('.pt')]
        if len(mask_files) >= 10:
            checks.append(("界面Maskfile", "✅", f"{len(mask_files)}个file"))
        else:
            checks.append(("界面Maskfile", "⚠️", f"只有{len(mask_files)}个file（期望>=10）"))
    else:
        checks.append(("界面Maskfile", "❌", "directory不Exists"))
    
    # CheckData集Split
    split_dir = "data/splits"
    if os.path.exists(split_dir):
        for split_file in ['train.txt', 'val.txt', 'test.txt']:
            split_path = os.path.join(split_dir, split_file)
            if os.path.exists(split_path):
                with open(split_path, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                checks.append((f"Data集Split {split_file}", "✅", f"{len(lines)}个样本"))
            else:
                checks.append((f"Data集Split {split_file}", "❌", "file不Exists"))
    else:
        checks.append(("Data集Split", "❌", "directory不Exists"))
    
    # 打印Check结果
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    all_passed = all('❌' not in c[1] for c in checks)
    print()
    return all_passed


def check_training():
    """CheckTrain结果是否完整"""
    print("="*60)
    print("CheckTrain结果")
    print("="*60)
    
    checks = []
    
    # Checkcheckpointdirectory
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        # Check最佳Model
        best_model = "checkpoints/best_complexmpnn.pt"
        if os.path.exists(best_model):
            size_mb = os.path.getsize(best_model) / (1024 * 1024)
            checks.append(("best_model checkpoint", "✅", f"{size_mb:.2f} MB"))
            
            # 尝试loadModel
            try:
                model_state = torch.load(best_model, map_location='cpu', weights_only=False)
                if isinstance(model_state, dict):
                    param_count = sum(v.numel() for v in model_state.values())
                    checks.append(("ModelParameterload", "✅", f"{param_count:,} 个Parameter"))
            except Exception as e:
                checks.append(("ModelParameterload", "❌", str(e)))
        else:
            checks.append(("best_model checkpoint", "❌", "file不Exists"))
        
        # Check定期checkpoint
        epoch_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('complexmpnn_epoch_') and f.endswith('.pt')]
        if epoch_files:
            checks.append(("定期 checkpoint", "✅", f"{len(epoch_files)} 个epochfile"))
        else:
            checks.append(("定期 checkpoint", "⚠️", "未找到epochfile"))
    else:
        checks.append(("Checkpointdirectory", "❌", "directory不Exists"))
    
    # 打印Check结果
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    all_passed = all('❌' not in c[1] for c in checks)
    print()
    return all_passed


def check_evaluation():
    """CheckEvaluate结果是否完整"""
    print("="*60)
    print("CheckEvaluate结果")
    print("="*60)
    
    checks = []
    
    # CheckEvaluatedirectory
    eval_dir = "logs/evaluation"
    if os.path.exists(eval_dir):
        # Checksequence_recovery_results.pt
        recovery_pt = "logs/evaluation/sequence_recovery_results.pt"
        if os.path.exists(recovery_pt):
            checks.append(("sequence恢复结果", "✅", "fileExists"))
            try:
                results = torch.load(recovery_pt, map_location='cpu', weights_only=False)
                if 'complex_mpnn' in results and 'baseline' in results:
                    checks.append(("sequence恢复Data", "✅", "格式正确"))
            except Exception as e:
                checks.append(("sequence恢复Data", "❌", str(e)))
        else:
            checks.append(("sequence恢复结果", "❌", "file不Exists"))
        
        # CheckCSVfile
        csv_files = ['sequence_recovery_results.csv', 'af_multimer_results.csv', 'combined_evaluation_results.csv']
        for csv_file in csv_files:
            csv_path = os.path.join(eval_dir, csv_file)
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    checks.append((f"CSVfile {csv_file}", "✅", f"{len(df)} 行"))
                except Exception as e:
                    checks.append((f"CSVfile {csv_file}", "❌", str(e)))
            else:
                checks.append((f"CSVfile {csv_file}", "❌", "file不Exists"))
        
        # Check可视化图表
        png_files = ['sequence_recovery_comparison.png', 'af_multimer_metrics.png']
        for png_file in png_files:
            png_path = os.path.join(eval_dir, png_file)
            if os.path.exists(png_path):
                size_kb = os.path.getsize(png_path) / 1024
                checks.append((f"可视化 {png_file}", "✅", f"{size_kb:.1f} KB"))
            else:
                checks.append((f"可视化 {png_file}", "⚠️", "file不Exists"))
    else:
        checks.append(("Evaluatedirectory", "❌", "directory不Exists"))
    
    # 打印Check结果
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    all_passed = all('❌' not in c[1] for c in checks)
    print()
    return all_passed


def check_logs():
    """Check日志file"""
    print("="*60)
    print("Check日志file")
    print("="*60)
    
    checks = []
    
    # Check主要日志
    log_files = ['preprocess.log', 'train.log', 'evaluate.log', 'full_pipeline.log', 'evaluation_test.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            size_kb = os.path.getsize(log_file) / 1024
            checks.append((f"日志 {log_file}", "✅", f"{size_kb:.1f} KB"))
        else:
            checks.append((f"日志 {log_file}", "⚠️", "file不Exists"))
    
    # 打印Check结果
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    print()
    return True


def print_summary(prep_ok, train_ok, eval_ok, logs_ok):
    """打印总结"""
    print("="*60)
    print("全流程Verify总结")
    print("="*60)
    
    print(f"预ProcessingData:   {'✅ Passed' if prep_ok else '❌ Failed'}")
    print(f"Train结果:     {'✅ Passed' if train_ok else '❌ Failed'}")
    print(f"Evaluate结果:     {'✅ Passed' if eval_ok else '❌ Failed'}")
    print(f"日志file:     {'✅ CheckComplete' if logs_ok else '⚠️ 部分缺失'}")
    
    print()
    if prep_ok and train_ok and eval_ok:
        print("🎉 AllCheckPassed！全流程VerifySuccess！")
        return True
    else:
        print("⚠️ 部分Check未Passed，请Check上述output")
        return False


def main():
    """主Function"""
    print("ComplexMPNN 全流程Verify")
    print("Verify一键运行后Alloutputfile是否完整且符合要求")
    print()
    
    # 运行各项Check
    prep_ok = check_preprocessing()
    train_ok = check_training()
    eval_ok = check_evaluation()
    logs_ok = check_logs()
    
    # 打印总结
    success = print_summary(prep_ok, train_ok, eval_ok, logs_ok)
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
