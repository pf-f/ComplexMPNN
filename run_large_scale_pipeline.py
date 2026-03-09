#!/usr/bin/env python3
"""
大规模Train全流程自动化脚本
从PDBDownload到TrainComplete
"""

import os
import sys
import subprocess
import time


def run_command(cmd, description, log_file=None):
    """运行命令并记录日志"""
    print(f"\n{'='*60}")
    print(f"Start: {description}")
    print(f"{'='*60}")
    
    if log_file:
        cmd_with_log = f"{cmd} 2>&1 | tee -a {log_file}"
    else:
        cmd_with_log = cmd
    
    try:
        result = subprocess.run(cmd_with_log, shell=True, cwd='/home/pff-bio/pf-f/ComplexMPNN')
        if result.returncode == 0:
            print(f"✓ Complete: {description}")
            return True
        else:
            print(f"✗ Failed: {description}")
            return False
    except Exception as e:
        print(f"✗ Error: {description} - {str(e)}")
        return False


def main():
    # 创建日志directory
    os.makedirs('logs', exist_ok=True)
    main_log = 'logs/large_scale_pipeline.log'
    
    print("="*60)
    print("ComplexMPNN 大规模Train全流程")
    print("="*60)
    
    # 步骤1: DownloadPDB（先Download500个测试）
    step1 = run_command(
        "python fetch_large_scale_assemblies.py --pdb_list large_pdb_ids.txt --output_dir data/raw_pdb --max_workers 15 --limit 500",
        "Download500个PDBfile",
        main_log
    )
    if not step1:
        print("DownloadFailed，停止执行")
        return 1
    
    # 步骤2: Filterheteromeric complexes
    step2 = run_command(
        "python filter_heteromeric_complexes.py --input_dir data/raw_pdb --output_dir data/processed/structures",
        "Filterheteromeric complexes",
        main_log
    )
    if not step2:
        print("FilterFailed，停止执行")
        return 1
    
    # 步骤3: Detect界面
    step3 = run_command(
        "python detect_interfaces.py --input_dir data/processed/structures --output_dir data/processed/interface_masks",
        "Detect蛋白质界面",
        main_log
    )
    if not step3:
        print("界面DetectFailed，停止执行")
        return 1
    
    # 步骤4: BuildMPNNfile
    step4 = run_command(
        "python build_mpnn_pt_files.py --input_dir data/processed/structures --interface_dir data/processed/interface_masks --output_dir data/processed/mpnn_pt",
        "BuildMPNNTrainfile",
        main_log
    )
    if not step4:
        print("MPNNfileBuildFailed，停止执行")
        return 1
    
    # 步骤5: Cluster和Split
    step5 = run_command(
        "python cluster_and_split.py --input_dir data/processed/mpnn_pt --output_dir data/splits",
        "Cluster和数据集Split",
        main_log
    )
    if not step5:
        print("ClusterSplitFailed，停止执行")
        return 1
    
    # 步骤6: Train模型（GPU）
    print("\n" + "="*60)
    print("StartGPUTrain...")
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
        "GPUTrain模型",
        main_log
    )
    
    print("\n" + "="*60)
    if step6:
        print("✓ 全流程Complete！")
        print("模型save在 checkpoints/")
    else:
        print("✗ TrainFailed")
    print("="*60)
    
    return 0 if step6 else 1


if __name__ == "__main__":
    sys.exit(main())
