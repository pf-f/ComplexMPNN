#!/usr/bin/env python3
"""
detect_interfaces.py

Function：Detectprotein complexes中的interface residues

Core Rules：
1. interface residues定义为Cβ-Cβ<8Å（Gly用Cα）
2. outputnpy格式的interface_mask

Dependencies：
- biopython
- numpy

Usage：
python detect_interfaces.py --input_dir filtered_complexes --output_dir data/processed/interface_masks
"""

import os
import argparse
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

# 忽略PDB结构BuildWarning
warnings.filterwarnings('ignore', category=PDBConstructionWarning)


def get_atom_coordinates(residue):
    """
    获取residue的Cβ原子Coordinates（Gly使用Cα）
    
    Args:
        residue: PDBresidue对象
    
    Returns:
        原子Coordinates（x, y, z），如果没有则返回None
    """
    try:
        if residue.get_resname() == 'GLY':
            # Gly使用Cα
            atom = residue['CA']
        else:
            # 其他amino acid使用Cβ
            atom = residue['CB']
        return atom.get_coord()
    except KeyError:
        # 如果原子不Exists，返回None
        return None


def calculate_distance(coord1, coord2):
    """
    计算两个原子Coordinates之间的距离
    
    Args:
        coord1: 第一个原子Coordinates
        coord2: 第二个原子Coordinates
    
    Returns:
        距离值
    """
    return np.sqrt(np.sum((coord1 - coord2) ** 2))


def detect_interface(structure):
    """
    Detect复合物中的interface residues
    
    Args:
        structure: PDB结构对象
    
    Returns:
        字典，键为chainID，值为interface residuesMask（numpy数组）
    """
    # 获取Allchain
    chains = list(structure.get_chains())
    num_chains = len(chains)
    
    # 如果chain数小于2，返回空字典
    if num_chains < 2:
        return {}
    
    # 为每个chainCreateresidue列表和Coordinates列表
    chain_residues = {}
    chain_coords = {}
    
    for chain in chains:
        residues = []
        coords = []
        for residue in chain:
            # Exclude水和配体
            if residue.id[0] == ' ' and residue.get_resname() not in ['HOH', 'WAT']:
                residues.append(residue)
                coord = get_atom_coordinates(residue)
                coords.append(coord)
        chain_residues[chain.id] = residues
        chain_coords[chain.id] = coords
    
    # Detectinterface residues
    interface_masks = {}
    
    for i, chain1 in enumerate(chains):
        chain1_id = chain1.id
        chain1_residues = chain_residues[chain1_id]
        chain1_coords = chain_coords[chain1_id]
        
        # 初始化界面Mask
        mask = np.zeros(len(chain1_residues), dtype=bool)
        
        # Check与其他chain的距离
        for j, chain2 in enumerate(chains):
            if i == j:
                continue
            
            chain2_id = chain2.id
            chain2_coords = chain_coords[chain2_id]
            
            # 计算chain1中每个residue与chain2中Allresidue的距离
            for k, coord1 in enumerate(chain1_coords):
                if coord1 is None:
                    continue
                
                for coord2 in chain2_coords:
                    if coord2 is None:
                        continue
                    
                    distance = calculate_distance(coord1, coord2)
                    if distance < 8.0:
                        mask[k] = True
                        break  # 找到一个满足条件的residue就可以停止
        
        interface_masks[chain1_id] = mask
    
    return interface_masks


def main():
    """
    主Function
    """
    parser = argparse.ArgumentParser(description='Detectprotein complexes中的interface residues')
    parser.add_argument('--input_dir', required=True, help='inputPDBfiledirectory')
    parser.add_argument('--output_dir', default='data/processed/interface_masks', help='outputdirectory')
    
    args = parser.parse_args()
    
    # 确保outputdirectoryExists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取inputdirectory中的PDBfile
    pdb_files = []
    for file in os.listdir(args.input_dir):
        if file.endswith('.pdb') or file.endswith('.cif'):
            pdb_files.append(os.path.join(args.input_dir, file))
    
    print(f"StartProcessing {len(pdb_files)} 个PDBfile")
    
    # Processing每个PDBfile
    for pdb_file in pdb_files:
        try:
            # 解析PDBfile
            if pdb_file.endswith('.cif'):
                parser = MMCIFParser()
            else:
                parser = PDBParser()
            structure = parser.get_structure('structure', pdb_file)
            
            # Detectinterface residues
            interface_masks = detect_interface(structure)
            
            # save界面Mask
            pdb_id = os.path.basename(pdb_file).split('_')[0]  # 假设file名格式为 {pdb_id}_assembly1.pdb
            
            for chain_id, mask in interface_masks.items():
                output_file = os.path.join(args.output_dir, f"{pdb_id}_{chain_id}.npy")
                np.save(output_file, mask)
                print(f"已save {output_file}")
        except Exception as e:
            print(f"Processing {pdb_file} 时出错：{str(e)}")
    
    print("界面DetectComplete")


if __name__ == "__main__":
    main()
