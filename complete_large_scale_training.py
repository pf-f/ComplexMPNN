#!/usr/bin/env python3
"""
完整的大规模Train自动化脚本
确保所有步骤都能Complete
"""

import os
import sys
import subprocess
import time


def run_step(cmd, step_name, log_file):
    """运行一个步骤并记录"""
    print(f"\n{'='*70}")
    print(f"Start: {step_name}")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    full_cmd = f"{cmd} 2>&1 | tee -a {log_file}"
    
    try:
        result = subprocess.run(full_cmd, shell=True, cwd='/home/pff-bio/pf-f/ComplexMPNN')
        if result.returncode == 0:
            print(f"✓ Success: {step_name}")
            return True
        else:
            print(f"✗ Failed: {step_name}")
            return False
    except Exception as e:
        print(f"✗ Error: {step_name} - {str(e)}")
        return False


def main():
    # 创建日志directory
    os.makedirs('logs', exist_ok=True)
    main_log = 'logs/complete_large_scale_training.log'
    
    print("="*70)
    print("ComplexMPNN 完整大规模Train流程")
    print("="*70)
    
    # 步骤1: Detect界面
    step1 = run_step(
        "python detect_interfaces.py --input_dir data/processed/structures --output_dir data/processed/interface_masks",
        "Detect蛋白质界面",
        main_log
    )
    if not step1:
        print("\n界面DetectFailed，但尝试Continue...")
    
    # 步骤2: BuildMPNNfile
    step2 = run_step(
        "python build_mpnn_pt_files.py --input_dir data/processed/structures --interface_dir data/processed/interface_masks --output_dir data/processed/mpnn_pt",
        "BuildMPNNTrainfile",
        main_log
    )
    if not step2:
        print("\nMPNNfileBuildFailed，但尝试Continue...")
    
    # 步骤3: Cluster和Split
    step3 = run_step(
        "python cluster_and_split.py --input_dir data/processed/mpnn_pt --output_dir data/splits",
        "Cluster和数据集Split",
        main_log
    )
    if not step3:
        print("\nClusterSplitFailed，但尝试Continue...")
    
    # 步骤4: 清理旧checkpoints
    print("\n清理旧checkpoint...")
    import glob
    for f in glob.glob('checkpoints/*.pt'):
        try:
            os.remove(f)
        except:
            pass
    
    # 步骤5: GPUTrain
    print("\n" + "="*70)
    print("StartGPUTrain（30 epochs）")
    print("="*70)
    sys.stdout.flush()
    
    step5 = run_step(
        "python train_complex_mpnn.py --config config.yaml",
        "GPUTrain模型",
        main_log
    )
    
    # 总结
    print("\n" + "="*70)
    if step5:
        print("✓ 全流程Complete！")
        print("模型save在 checkpoints/")
        print("best_model: checkpoints/best_complexmpnn.pt")
    else:
        print("✗ Train可能Failed，请检查日志")
    print("="*70)
    
    return 0 if step5 else 1


if __name__ == "__main__":
    sys.exit(main())
