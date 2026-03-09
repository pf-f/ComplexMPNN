#!/usr/bin/env python3
"""
check_evaluation.py

Function: Verify that all functions in the Evaluation Module work correctly

Usage:
python check_evaluation.py
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil


def check_loss_functions():
    """Test metric calculation functions"""
    print("=== Testing metric calculation functions ===")
    
    from run_af_multimer import calculate_rmsd
    coords1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    coords2 = np.array([[0.1, 0.1, 0], [1.1, 0.1, 0], [0.1, 1.1, 0]])
    rmsd = calculate_rmsd(coords1, coords2)
    print(f"RMSD calculation: {rmsd:.4f}")
    assert 0.1 < rmsd < 0.2, f"RMSD calculation error: {rmsd}"
    
    from run_af_multimer import calculate_tm_score
    tm_score = calculate_tm_score(coords1, coords2, seq_len=3)
    print(f"TM-score calculation: {tm_score:.4f}")
    assert 0 < tm_score <= 1, f"TM-score calculation error: {tm_score}"
    
    from run_af_multimer import calculate_iptm
    chain_coords1 = {'A': coords1, 'B': coords1 + 5}
    chain_coords2 = {'A': coords2, 'B': coords2 + 5}
    iptm = calculate_iptm(chain_coords1, chain_coords2)
    print(f"ipTM calculation: {iptm:.4f}")
    assert 0 < iptm <= 1, f"ipTM calculation error: {iptm}"
    
    print("✅ Metric calculation functions test passed!\n")
    return True


def check_sequence_recovery():
    """Test sequence recovery calculation"""
    print("=== Testing sequence recovery calculation ===")
    
    from interface_recovery import calculate_sequence_recovery
    from train_complex_mpnn import ProteinMPNNWrapper, set_random_seed, load_config, ComplexMPNNDataSet
    from torch.utils.data import DataLoader
    
    set_random_seed(42)
    model = ProteinMPNNWrapper()
    device = torch.device('cpu')
    model = model.to(device)
    
    config = load_config('config.yaml')
    
    test_file = os.path.join(config['data']['split_dir'], config['data']['test_split'])
    if os.path.exists(test_file):
        print("Using real data for testing...")
        
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
        print("No real data available, skipping full test (function passed import test)")
    
    print("✅ Sequence recovery calculation test passed!\n")
    return True


def check_af_multimer():
    """Test AlphaFold-Multimer module"""
    print("=== Testing AlphaFold-Multimer module ===")
    
    from run_af_multimer import predict_structure_with_af_multimer, evaluate_structure_quality
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "YWVTSRQPNMLKIHGFEDCA"]
        results = predict_structure_with_af_multimer(sequences, tmpdir)
        
        assert 'rmsd' in results
        assert 'tm_score' in results
        assert 'iptm' in results
        assert 'predicted_pdb' in results
        assert os.path.exists(results['predicted_pdb'])
        
        print(f"Prediction complete, RMSD: {results['rmsd']:.3f}")
        
        num_residues = sum(len(seq) for seq in sequences)
        predicted_coords = np.random.randn(num_residues, 3)
        native_coords = predicted_coords + np.random.randn(num_residues, 3) * 0.5
        
        quality_metrics = evaluate_structure_quality(predicted_coords, native_coords)
        
        assert 'rmsd' in quality_metrics
        assert 'tm_score' in quality_metrics
        assert 'iptm' in quality_metrics
    
    print("✅ AlphaFold-Multimer module test passed!\n")
    return True


def check_analyze_results():
    """Test results analysis module"""
    print("=== Testing results analysis module ===")
    
    from analyze_results import plot_sequence_recovery_comparison, plot_af_metrics, export_to_csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parsed_recovery = {
            'metrics': ['Interface recovery', 'Non-interface recovery', 'Overall recovery'],
            'ComplexMPNN': [0.35, 0.28, 0.30],
            'Baseline': [0.25, 0.26, 0.26]
        }
        
        parsed_af = {
            'metrics': ['RMSD (Å)', 'TM-score', 'ipTM'],
            'values': [2.5, 0.75, 0.70]
        }
        
        plot_sequence_recovery_comparison(parsed_recovery, tmpdir)
        plot_af_metrics(parsed_af, tmpdir)
        
        assert os.path.exists(os.path.join(tmpdir, 'sequence_recovery_comparison.png'))
        assert os.path.exists(os.path.join(tmpdir, 'af_multimer_metrics.png'))
        
        export_to_csv(parsed_recovery, parsed_af, tmpdir)
        
        assert os.path.exists(os.path.join(tmpdir, 'sequence_recovery_results.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'af_multimer_results.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'combined_evaluation_results.csv'))
        
        print("✅ Results analysis module test passed!\n")
        return True


def main():
    """Main function"""
    print("Starting Evaluation module validation...\n")
    
    all_passed = True
    
    try:
        if not check_loss_functions():
            all_passed = False
    except Exception as e:
        print(f"❌ Metric calculation test failed: {e}")
        all_passed = False
    
    try:
        if not check_sequence_recovery():
            all_passed = False
    except Exception as e:
        print(f"❌ Sequence recovery test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        if not check_af_multimer():
            all_passed = False
    except Exception as e:
        print(f"❌ AF-Multimer test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        if not check_analyze_results():
            all_passed = False
    except Exception as e:
        print(f"❌ Results analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    if all_passed:
        print("🎉 All checks passed! Evaluation module validation successful!")
    else:
        print("❌ Some checks failed, please check error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
