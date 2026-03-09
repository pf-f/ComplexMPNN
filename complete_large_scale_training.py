#!/usr/bin/env python3
"""
完整的大规模训练自动化脚本
确保所有步骤都能完成
"""

import os
import sys
import subprocess
import time


def run_step(cmd, step_name, log_file):
    """运行一个步骤并记录"""
    print(f"\n{'='*70}")
    print(f"开始: {step_name}")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    full_cmd = f"{cmd} 2>&1 | tee -a {log_file}"
    
    try:
        result = subprocess.run(full_cmd, shell=True, cwd='/home/pff-bio/pf-f/ComplexMPNN')
        if result.returncode == 0:
            print(f"✓ 成功: {step_name}")
            return True
        else:
            print(f"✗ 失败: {step_name}")
            return False
    except Exception as e:
        print(f"✗ 错误: {step_name} - {str(e)}")
        return False


def main():
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    main_log = 'logs/complete_large_scale_training.log'
    
    print("="*70)
    print("ComplexMPNN 完整大规模训练流程")
    print("="*70)
    
    # 步骤1: 检测界面
    step1 = run_step(
        "python detect_interfaces.py --input_dir data/processed/structures --output_dir data/processed/interface_masks",
        "检测蛋白质界面",
        main_log
    )
    if not step1:
        print("\n界面检测失败，但尝试继续...")
    
    # 步骤2: 构建MPNN文件
    step2 = run_step(
        "python build_mpnn_pt_files.py --input_dir data/processed/structures --interface_dir data/processed/interface_masks --output_dir data/processed/mpnn_pt",
        "构建MPNN训练文件",
        main_log
    )
    if not step2:
        print("\nMPNN文件构建失败，但尝试继续...")
    
    # 步骤3: 聚类和切分
    step3 = run_step(
        "python cluster_and_split.py --input_dir data/processed/mpnn_pt --output_dir data/splits",
        "聚类和数据集切分",
        main_log
    )
    if not step3:
        print("\n聚类切分失败，但尝试继续...")
    
    # 步骤4: 清理旧checkpoints
    print("\n清理旧检查点...")
    import glob
    for f in glob.glob('checkpoints/*.pt'):
        try:
            os.remove(f)
        except:
            pass
    
    # 步骤5: GPU训练
    print("\n" + "="*70)
    print("开始GPU训练（30 epochs）")
    print("="*70)
    sys.stdout.flush()
    
    step5 = run_step(
        "python train_complex_mpnn.py --config config.yaml",
        "GPU训练模型",
        main_log
    )
    
    # 总结
    print("\n" + "="*70)
    if step5:
        print("✓ 全流程完成！")
        print("模型保存在 checkpoints/")
        print("最佳模型: checkpoints/best_complexmpnn.pt")
    else:
        print("✗ 训练可能失败，请检查日志")
    print("="*70)
    
    return 0 if step5 else 1


if __name__ == "__main__":
    sys.exit(main())
