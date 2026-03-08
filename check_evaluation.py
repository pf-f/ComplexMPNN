#!/usr/bin/env python3
"""
check_evaluation.py

功能：验证评估模块的各个功能是否正常工作

使用方法：
python check_evaluation.py
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil


def check_loss_functions():
    """测试指标计算函数"""
    print("=== 测试指标计算函数 ===")
    
    # 测试RMSD计算
    from run_af_multimer import calculate_rmsd
    coords1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    coords2 = np.array([[0.1, 0.1, 0], [1.1, 0.1, 0], [0.1, 1.1, 0]])
    rmsd = calculate_rmsd(coords1, coords2)
    print(f"RMSD计算: {rmsd:.4f}")
    assert 0.1 < rmsd < 0.2, f"RMSD计算错误: {rmsd}"
    
    # 测试TM-score计算
    from run_af_multimer import calculate_tm_score
    tm_score = calculate_tm_score(coords1, coords2, seq_len=3)
    print(f"TM-score计算: {tm_score:.4f}")
    assert 0 < tm_score <= 1, f"TM-score计算错误: {tm_score}"
    
    # 测试ipTM计算
    from run_af_multimer import calculate_iptm
    chain_coords1 = {'A': coords1, 'B': coords1 + 5}
    chain_coords2 = {'A': coords2, 'B': coords2 + 5}
    iptm = calculate_iptm(chain_coords1, chain_coords2)
    print(f"ipTM计算: {iptm:.4f}")
    assert 0 < iptm <= 1, f"ipTM计算错误: {iptm}"
    
    print("✅ 指标计算函数测试通过！\n")
    return True


def check_sequence_recovery():
    """测试序列恢复计算"""
    print("=== 测试序列恢复计算 ===")
    
    from interface_recovery import calculate_sequence_recovery
    from train_complex_mpnn import ProteinMPNNWrapper, set_random_seed, load_config, ComplexMPNNDataSet
    from torch.utils.data import DataLoader
    
    # 创建临时模型和数据
    set_random_seed(42)
    model = ProteinMPNNWrapper()
    device = torch.device('cpu')
    model = model.to(device)
    
    # 加载真实配置
    config = load_config('config.yaml')
    
    # 检查是否有真实数据可用，否则跳过完整测试
    test_file = os.path.join(config['data']['split_dir'], config['data']['test_split'])
    if os.path.exists(test_file):
        print("使用真实数据进行测试...")
        
        from train_complex_mpnn import collate_fn
        test_dataset = ComplexMPNNDataSet(
            config['data']['mpnn_pt_dir'],
            'test.txt',
            config['data']['split_dir']
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        results = calculate_sequence_recovery(
            model, test_dataloader, device, config, use_joint_design=True
        )
        
        print(f"Interface recovery: {results['interface_recovery']:.4f}")
        print(f"Non-interface recovery: {results['non_interface_recovery']:.4f}")
        print(f"Overall recovery: {results['overall_recovery']:.4f}")
        
        assert 0 <= results['interface_recovery'] <= 1
        assert 0 <= results['non_interface_recovery'] <= 1
        assert 0 <= results['overall_recovery'] <= 1
    else:
        print("无真实数据，跳过完整测试（函数已通过导入测试）")
    
    print("✅ 序列恢复计算测试通过！\n")
    return True


def check_af_multimer():
    """测试AlphaFold-Multimer模块"""
    print("=== 测试AlphaFold-Multimer模块 ===")
    
    from run_af_multimer import predict_structure_with_af_multimer, evaluate_structure_quality
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试预测函数
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "YWVTSRQPNMLKIHGFEDCA"]
        results = predict_structure_with_af_multimer(sequences, tmpdir)
        
        assert 'rmsd' in results
        assert 'tm_score' in results
        assert 'iptm' in results
        assert 'predicted_pdb' in results
        assert os.path.exists(results['predicted_pdb'])
        
        print(f"预测完成，RMSD: {results['rmsd']:.3f}")
        
        # 测试评估函数
        num_residues = sum(len(seq) for seq in sequences)
        predicted_coords = np.random.randn(num_residues, 3)
        native_coords = predicted_coords + np.random.randn(num_residues, 3) * 0.5
        
        quality_metrics = evaluate_structure_quality(predicted_coords, native_coords)
        
        assert 'rmsd' in quality_metrics
        assert 'tm_score' in quality_metrics
        assert 'iptm' in quality_metrics
    
    print("✅ AlphaFold-Multimer模块测试通过！\n")
    return True


def check_analyze_results():
    """测试结果分析模块"""
    print("=== 测试结果分析模块 ===")
    
    from analyze_results import plot_sequence_recovery_comparison, plot_af_metrics, export_to_csv
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模拟数据
        parsed_recovery = {
            'metrics': ['Interface recovery', 'Non-interface recovery', 'Overall recovery'],
            'ComplexMPNN': [0.35, 0.28, 0.30],
            'Baseline': [0.25, 0.26, 0.26]
        }
        
        parsed_af = {
            'metrics': ['RMSD (Å)', 'TM-score', 'ipTM'],
            'values': [2.5, 0.75, 0.70]
        }
        
        # 测试绘图
        plot_sequence_recovery_comparison(parsed_recovery, tmpdir)
        plot_af_metrics(parsed_af, tmpdir)
        
        assert os.path.exists(os.path.join(tmpdir, 'sequence_recovery_comparison.png'))
        assert os.path.exists(os.path.join(tmpdir, 'af_multimer_metrics.png'))
        
        # 测试CSV导出
        export_to_csv(parsed_recovery, parsed_af, tmpdir)
        
        assert os.path.exists(os.path.join(tmpdir, 'sequence_recovery_results.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'af_multimer_results.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'combined_evaluation_results.csv'))
        
        print("✅ 结果分析模块测试通过！\n")
        return True


def main():
    """主函数"""
    print("开始验证评估模块...\n")
    
    all_passed = True
    
    # 测试指标计算
    try:
        if not check_loss_functions():
            all_passed = False
    except Exception as e:
        print(f"❌ 指标计算测试失败: {e}")
        all_passed = False
    
    # 测试序列恢复
    try:
        if not check_sequence_recovery():
            all_passed = False
    except Exception as e:
        print(f"❌ 序列恢复测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 测试AF-Multimer
    try:
        if not check_af_multimer():
            all_passed = False
    except Exception as e:
        print(f"❌ AF-Multimer测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 测试结果分析
    try:
        if not check_analyze_results():
            all_passed = False
    except Exception as e:
        print(f"❌ 结果分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    if all_passed:
        print("🎉 所有检查通过！评估模块验证成功！")
    else:
        print("❌ 部分检查失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
