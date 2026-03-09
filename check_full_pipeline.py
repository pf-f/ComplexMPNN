#!/usr/bin/env python3
"""
check_full_pipeline.py

Function：验证一键运行后所有outputfile是否完整且符合要求

Usage：
python check_full_pipeline.py
"""

import os
import sys
import torch
import pandas as pd


def check_preprocessing():
    """检查预Processing数据是否完整"""
    print("="*60)
    print("检查预Processing数据")
    print("="*60)
    
    checks = []
    
    # 检查mpnn_ptdirectory
    mpnn_pt_dir = "data/processed/mpnn_pt"
    if os.path.exists(mpnn_pt_dir):
        pt_files = [f for f in os.listdir(mpnn_pt_dir) if f.endswith('.pt')]
        if len(pt_files) >= 10:
            checks.append(("MPNN .ptfile", "✅", f"{len(pt_files)}个file"))
        else:
            checks.append(("MPNN .ptfile", "⚠️", f"只有{len(pt_files)}个file（期望>=10）"))
    else:
        checks.append(("MPNN .ptfile", "❌", "directory不存在"))
    
    # 检查interface_masksdirectory
    interface_masks_dir = "data/processed/interface_masks"
    if os.path.exists(interface_masks_dir):
        mask_files = [f for f in os.listdir(interface_masks_dir) if f.endswith('.pt')]
        if len(mask_files) >= 10:
            checks.append(("界面掩码file", "✅", f"{len(mask_files)}个file"))
        else:
            checks.append(("界面掩码file", "⚠️", f"只有{len(mask_files)}个file（期望>=10）"))
    else:
        checks.append(("界面掩码file", "❌", "directory不存在"))
    
    # 检查数据集Split
    split_dir = "data/splits"
    if os.path.exists(split_dir):
        for split_file in ['train.txt', 'val.txt', 'test.txt']:
            split_path = os.path.join(split_dir, split_file)
            if os.path.exists(split_path):
                with open(split_path, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                checks.append((f"数据集Split {split_file}", "✅", f"{len(lines)}个样本"))
            else:
                checks.append((f"数据集Split {split_file}", "❌", "file不存在"))
    else:
        checks.append(("数据集Split", "❌", "directory不存在"))
    
    # 打印检查结果
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    all_passed = all('❌' not in c[1] for c in checks)
    print()
    return all_passed


def check_training():
    """检查Train结果是否完整"""
    print("="*60)
    print("检查Train结果")
    print("="*60)
    
    checks = []
    
    # 检查checkpointdirectory
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        # 检查最佳模型
        best_model = "checkpoints/best_complexmpnn.pt"
        if os.path.exists(best_model):
            size_mb = os.path.getsize(best_model) / (1024 * 1024)
            checks.append(("best_model checkpoint", "✅", f"{size_mb:.2f} MB"))
            
            # 尝试load模型
            try:
                model_state = torch.load(best_model, map_location='cpu', weights_only=False)
                if isinstance(model_state, dict):
                    param_count = sum(v.numel() for v in model_state.values())
                    checks.append(("模型参数load", "✅", f"{param_count:,} 个参数"))
            except Exception as e:
                checks.append(("模型参数load", "❌", str(e)))
        else:
            checks.append(("best_model checkpoint", "❌", "file不存在"))
        
        # 检查定期checkpoint
        epoch_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('complexmpnn_epoch_') and f.endswith('.pt')]
        if epoch_files:
            checks.append(("定期 checkpoint", "✅", f"{len(epoch_files)} 个epochfile"))
        else:
            checks.append(("定期 checkpoint", "⚠️", "未找到epochfile"))
    else:
        checks.append(("Checkpointdirectory", "❌", "directory不存在"))
    
    # 打印检查结果
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    all_passed = all('❌' not in c[1] for c in checks)
    print()
    return all_passed


def check_evaluation():
    """检查Evaluate结果是否完整"""
    print("="*60)
    print("检查Evaluate结果")
    print("="*60)
    
    checks = []
    
    # 检查Evaluatedirectory
    eval_dir = "logs/evaluation"
    if os.path.exists(eval_dir):
        # 检查sequence_recovery_results.pt
        recovery_pt = "logs/evaluation/sequence_recovery_results.pt"
        if os.path.exists(recovery_pt):
            checks.append(("sequence恢复结果", "✅", "file存在"))
            try:
                results = torch.load(recovery_pt, map_location='cpu', weights_only=False)
                if 'complex_mpnn' in results and 'baseline' in results:
                    checks.append(("sequence恢复数据", "✅", "格式正确"))
            except Exception as e:
                checks.append(("sequence恢复数据", "❌", str(e)))
        else:
            checks.append(("sequence恢复结果", "❌", "file不存在"))
        
        # 检查CSVfile
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
                checks.append((f"CSVfile {csv_file}", "❌", "file不存在"))
        
        # 检查可视化图表
        png_files = ['sequence_recovery_comparison.png', 'af_multimer_metrics.png']
        for png_file in png_files:
            png_path = os.path.join(eval_dir, png_file)
            if os.path.exists(png_path):
                size_kb = os.path.getsize(png_path) / 1024
                checks.append((f"可视化 {png_file}", "✅", f"{size_kb:.1f} KB"))
            else:
                checks.append((f"可视化 {png_file}", "⚠️", "file不存在"))
    else:
        checks.append(("Evaluatedirectory", "❌", "directory不存在"))
    
    # 打印检查结果
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    all_passed = all('❌' not in c[1] for c in checks)
    print()
    return all_passed


def check_logs():
    """检查日志file"""
    print("="*60)
    print("检查日志file")
    print("="*60)
    
    checks = []
    
    # 检查主要日志
    log_files = ['preprocess.log', 'train.log', 'evaluate.log', 'full_pipeline.log', 'evaluation_test.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            size_kb = os.path.getsize(log_file) / 1024
            checks.append((f"日志 {log_file}", "✅", f"{size_kb:.1f} KB"))
        else:
            checks.append((f"日志 {log_file}", "⚠️", "file不存在"))
    
    # 打印检查结果
    for name, status, detail in checks:
        print(f"{status} {name}: {detail}")
    
    print()
    return True


def print_summary(prep_ok, train_ok, eval_ok, logs_ok):
    """打印总结"""
    print("="*60)
    print("全流程验证总结")
    print("="*60)
    
    print(f"预Processing数据:   {'✅ 通过' if prep_ok else '❌ Failed'}")
    print(f"Train结果:     {'✅ 通过' if train_ok else '❌ Failed'}")
    print(f"Evaluate结果:     {'✅ 通过' if eval_ok else '❌ Failed'}")
    print(f"日志file:     {'✅ 检查Complete' if logs_ok else '⚠️ 部分缺失'}")
    
    print()
    if prep_ok and train_ok and eval_ok:
        print("🎉 所有检查通过！全流程验证Success！")
        return True
    else:
        print("⚠️ 部分检查未通过，请检查上述output")
        return False


def main():
    """主函数"""
    print("ComplexMPNN 全流程验证")
    print("验证一键运行后所有outputfile是否完整且符合要求")
    print()
    
    # 运行各项检查
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
