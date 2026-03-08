#!/usr/bin/env python3
"""
detect_interfaces.py

功能：检测蛋白质复合物中的界面残基

核心规则：
1. 界面残基定义为Cβ-Cβ<8Å（Gly用Cα）
2. 输出npy格式的interface_mask

依赖：
- biopython
- numpy

使用方法：
python detect_interfaces.py --input_dir filtered_complexes --output_dir data/processed/interface_masks
"""

import os
import argparse
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

# 忽略PDB结构构建警告
warnings.filterwarnings('ignore', category=PDBConstructionWarning)


def get_atom_coordinates(residue):
    """
    获取残基的Cβ原子坐标（Gly使用Cα）
    
    Args:
        residue: PDB残基对象
    
    Returns:
        原子坐标（x, y, z），如果没有则返回None
    """
    try:
        if residue.get_resname() == 'GLY':
            # Gly使用Cα
            atom = residue['CA']
        else:
            # 其他氨基酸使用Cβ
            atom = residue['CB']
        return atom.get_coord()
    except KeyError:
        # 如果原子不存在，返回None
        return None


def calculate_distance(coord1, coord2):
    """
    计算两个原子坐标之间的距离
    
    Args:
        coord1: 第一个原子坐标
        coord2: 第二个原子坐标
    
    Returns:
        距离值
    """
    return np.sqrt(np.sum((coord1 - coord2) ** 2))


def detect_interface(structure):
    """
    检测复合物中的界面残基
    
    Args:
        structure: PDB结构对象
    
    Returns:
        字典，键为链ID，值为界面残基掩码（numpy数组）
    """
    # 获取所有链
    chains = list(structure.get_chains())
    num_chains = len(chains)
    
    # 如果链数小于2，返回空字典
    if num_chains < 2:
        return {}
    
    # 为每个链创建残基列表和坐标列表
    chain_residues = {}
    chain_coords = {}
    
    for chain in chains:
        residues = []
        coords = []
        for residue in chain:
            # 排除水和配体
            if residue.id[0] == ' ' and residue.get_resname() not in ['HOH', 'WAT']:
                residues.append(residue)
                coord = get_atom_coordinates(residue)
                coords.append(coord)
        chain_residues[chain.id] = residues
        chain_coords[chain.id] = coords
    
    # 检测界面残基
    interface_masks = {}
    
    for i, chain1 in enumerate(chains):
        chain1_id = chain1.id
        chain1_residues = chain_residues[chain1_id]
        chain1_coords = chain_coords[chain1_id]
        
        # 初始化界面掩码
        mask = np.zeros(len(chain1_residues), dtype=bool)
        
        # 检查与其他链的距离
        for j, chain2 in enumerate(chains):
            if i == j:
                continue
            
            chain2_id = chain2.id
            chain2_coords = chain_coords[chain2_id]
            
            # 计算链1中每个残基与链2中所有残基的距离
            for k, coord1 in enumerate(chain1_coords):
                if coord1 is None:
                    continue
                
                for coord2 in chain2_coords:
                    if coord2 is None:
                        continue
                    
                    distance = calculate_distance(coord1, coord2)
                    if distance < 8.0:
                        mask[k] = True
                        break  # 找到一个满足条件的残基就可以停止
        
        interface_masks[chain1_id] = mask
    
    return interface_masks


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='检测蛋白质复合物中的界面残基')
    parser.add_argument('--input_dir', required=True, help='输入PDB文件目录')
    parser.add_argument('--output_dir', default='data/processed/interface_masks', help='输出目录')
    
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
            # 解析PDB文件
            if pdb_file.endswith('.cif'):
                parser = MMCIFParser()
            else:
                parser = PDBParser()
            structure = parser.get_structure('structure', pdb_file)
            
            # 检测界面残基
            interface_masks = detect_interface(structure)
            
            # 保存界面掩码
            pdb_id = os.path.basename(pdb_file).split('_')[0]  # 假设文件名格式为 {pdb_id}_assembly1.pdb
            
            for chain_id, mask in interface_masks.items():
                output_file = os.path.join(args.output_dir, f"{pdb_id}_{chain_id}.npy")
                np.save(output_file, mask)
                print(f"已保存 {output_file}")
        except Exception as e:
            print(f"处理 {pdb_file} 时出错：{str(e)}")
    
    print("界面检测完成")


if __name__ == "__main__":
    main()
