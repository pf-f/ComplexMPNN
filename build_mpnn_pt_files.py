#!/usr/bin/env python3
"""
build_mpnn_pt_files.py

功能：将PDB文件转换为ProteinMPNN可直接训练的.pt文件

核心规则：
1. 使用ProteinMPNN官方的PDB parsing工具
2. 每条链生成一个.pt文件
3. 必须新增字段：interface_mask

依赖：
- biopython
- numpy
- torch

使用方法：
python build_mpnn_pt_files.py --input_dir filtered_complexes --interface_dir data/processed/interface_masks --output_dir data/processed/mpnn_pt
"""

import os
import argparse
import numpy as np
import torch
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

# 忽略PDB结构构建警告
warnings.filterwarnings('ignore', category=PDBConstructionWarning)


def parse_pdb_structure(pdb_file):
    """
    解析PDB文件结构
    
    Args:
        pdb_file: PDB文件路径
    
    Returns:
        PDB结构对象
    """
    try:
        if pdb_file.endswith('.cif'):
            parser = MMCIFParser()
        else:
            parser = PDBParser()
        structure = parser.get_structure('structure', pdb_file)
        return structure
    except Exception as e:
        print(f"解析 {pdb_file} 时出错：{str(e)}")
        return None


def get_chain_sequence(chain):
    """
    获取链的氨基酸序列
    
    Args:
        chain: PDB链对象
    
    Returns:
        氨基酸序列字符串
    """
    sequence = []
    aa_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    for residue in chain:
        if residue.id[0] == ' ' and residue.get_resname() not in ['HOH', 'WAT']:
            resname = residue.get_resname()
            if resname in aa_dict:
                sequence.append(aa_dict[resname])
    
    return ''.join(sequence)


def get_backbone_coords(chain):
    """
    获取链的主链原子坐标
    
    Args:
        chain: PDB链对象
    
    Returns:
        主链原子坐标数组 (N, 3, 3)，其中N是残基数，3是N、CA、C原子，3是坐标
    """
    coords = []
    
    for residue in chain:
        if residue.id[0] == ' ' and residue.get_resname() not in ['HOH', 'WAT']:
            try:
                n = residue['N'].get_coord()
                ca = residue['CA'].get_coord()
                c = residue['C'].get_coord()
                coords.append([n, ca, c])
            except KeyError:
                # 如果原子不存在，跳过
                continue
    
    return np.array(coords)


def build_mpnn_pt_file(pdb_file, interface_dir, output_dir):
    """
    构建MPNN .pt文件
    
    Args:
        pdb_file: PDB文件路径
        interface_dir: 界面掩码目录
        output_dir: 输出目录
    """
    # 解析PDB结构
    structure = parse_pdb_structure(pdb_file)
    if structure is None:
        return
    
    # 获取PDB ID
    pdb_id = os.path.basename(pdb_file).split('_')[0]
    
    # 处理每条链
    for chain in structure.get_chains():
        chain_id = chain.id
        
        # 获取序列
        sequence = get_chain_sequence(chain)
        if not sequence:
            continue
        
        # 获取主链坐标
        backbone_coords = get_backbone_coords(chain)
        if backbone_coords.shape[0] == 0:
            continue
        
        # 加载界面掩码
        interface_mask_file = os.path.join(interface_dir, f"{pdb_id}_{chain_id}.npy")
        if os.path.exists(interface_mask_file):
            interface_mask = np.load(interface_mask_file)
            # 确保掩码长度与序列长度匹配
            if len(interface_mask) != len(sequence):
                # 如果不匹配，创建长度为序列长度的全False掩码
                interface_mask = np.zeros(len(sequence), dtype=bool)
        else:
            # 如果没有界面掩码文件，创建全False掩码
            interface_mask = np.zeros(len(sequence), dtype=bool)
        
        # 创建MPNN数据字典
        data = {
            'sequence': sequence,
            'backbone_coords': torch.tensor(backbone_coords, dtype=torch.float32),
            'interface_mask': torch.tensor(interface_mask, dtype=torch.bool)
        }
        
        # 保存.pt文件
        output_file = os.path.join(output_dir, f"{pdb_id}_{chain_id}.pt")
        torch.save(data, output_file)
        print(f"已保存 {output_file}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='将PDB文件转换为ProteinMPNN可直接训练的.pt文件')
    parser.add_argument('--input_dir', required=True, help='输入PDB文件目录')
    parser.add_argument('--interface_dir', default='data/processed/interface_masks', help='界面掩码目录')
    parser.add_argument('--output_dir', default='data/processed/mpnn_pt', help='输出目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取输入目录中的PDB文件
    pdb_files = []
    for file in os.listdir(args.input_dir):
        if file.endswith('.pdb') or file.endswith('.cif'):
            pdb_files.append(os.path.join(args.input_dir, file))
    
    print(f"开始处理 {len(pdb_files)} 个PDB文件")
    
    # 处理每个PDB文件
    for pdb_file in pdb_files:
        try:
            build_mpnn_pt_file(pdb_file, args.interface_dir, args.output_dir)
        except Exception as e:
            print(f"处理 {pdb_file} 时出错：{str(e)}")
    
    print("MPNN .pt文件构建完成")


if __name__ == "__main__":
    main()
