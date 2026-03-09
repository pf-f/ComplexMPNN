#!/usr/bin/env python3
"""
大规模训练全流程自动化脚本
从PDB下载到训练完成
"""

import os
import sys
import subprocess
import time


def run_command(cmd, description, log_file=None):
    """运行命令并记录日志"""
    print(f"\n{'='*60}")
    print(f"开始: {description}")
    print(f"{'='*60}")
    
    if log_file:
        cmd_with_log = f"{cmd} 2>&1 | tee -a {log_file}"
    else:
        cmd_with_log = cmd
    
    try:
        result = subprocess.run(cmd_with_log, shell=True, cwd='/home/pff-bio/pf-f/ComplexMPNN')
        if result.returncode == 0:
            print(f"✓ 完成: {description}")
            return True
        else:
            print(f"✗ 失败: {description}")
            return False
    except Exception as e:
        print(f"✗ 错误: {description} - {str(e)}")
        return False


def main():
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    main_log = 'logs/large_scale_pipeline.log'
    
    print("="*60)
    print("ComplexMPNN 大规模训练全流程")
    print("="*60)
    
    # 步骤1: 下载PDB（先下载500个测试）
    step1 = run_command(
        "python fetch_large_scale_assemblies.py --pdb_list large_pdb_ids.txt --output_dir data/raw_pdb --max_workers 15 --limit 500",
        "下载500个PDB文件",
        main_log
    )
    if not step1:
        print("下载失败，停止执行")
        return 1
    
    # 步骤2: 过滤异源复合物
    step2 = run_command(
        "python filter_heteromeric_complexes.py --input_dir data/raw_pdb --output_dir data/processed/structures",
        "过滤异源复合物",
        main_log
    )
    if not step2:
        print("过滤失败，停止执行")
        return 1
    
    # 步骤3: 检测界面
    step3 = run_command(
        "python detect_interfaces.py --input_dir data/processed/structures --output_dir data/processed/interface_masks",
        "检测蛋白质界面",
        main_log
    )
    if not step3:
        print("界面检测失败，停止执行")
        return 1
    
    # 步骤4: 构建MPNN文件
    step4 = run_command(
        "python build_mpnn_pt_files.py --input_dir data/processed/structures --interface_dir data/processed/interface_masks --output_dir data/processed/mpnn_pt",
        "构建MPNN训练文件",
        main_log
    )
    if not step4:
        print("MPNN文件构建失败，停止执行")
        return 1
    
    # 步骤5: 聚类和切分
    step5 = run_command(
        "python cluster_and_split.py --input_dir data/processed/mpnn_pt --output_dir data/splits",
        "聚类和数据集切分",
        main_log
    )
    if not step5:
        print("聚类切分失败，停止执行")
        return 1
    
    # 步骤6: 训练模型（GPU）
    print("\n" + "="*60)
    print("开始GPU训练...")
    print("="*60)
    
    # 先清空旧的checkpoints
    import glob
    for f in glob.glob('checkpoints/*.pt'):
        try:
            os.remove(f)
        except:
            pass
    
    step6 = run_command(
        "python train_complex_mpnn.py --config config.yaml",
        "GPU训练模型",
        main_log
    )
    
    print("\n" + "="*60)
    if step6:
        print("✓ 全流程完成！")
        print("模型保存在 checkpoints/")
    else:
        print("✗ 训练失败")
    print("="*60)
    
    return 0 if step6 else 1


if __name__ == "__main__":
    sys.exit(main())
