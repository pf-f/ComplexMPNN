#!/usr/bin/env python3
"""
build_mpnn_pt_files.py

Function：将PDBfile转换为ProteinMPNN可直接Train的.ptfile

Core Rules：
1. 使用ProteinMPNN官方的PDB parsing工具
2. 每条chain生成一个.ptfile
3. 必须新增字段：interface_mask

Dependencies：
- biopython
- numpy
- torch

Usage：
python build_mpnn_pt_files.py --input_dir filtered_complexes --interface_dir data/processed/interface_masks --output_dir data/processed/mpnn_pt
"""

import os
import argparse
import numpy as np
import torch
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

# 忽略PDB结构BuildWarning
warnings.filterwarnings('ignore', category=PDBConstructionWarning)


def parse_pdb_structure(pdb_file):
    """
    解析PDBfile结构
    
    Args:
        pdb_file: PDBfilePath
    
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
    获取chain的amino acidsequence
    
    Args:
        chain: PDBchain对象
    
    Returns:
        amino acidsequence字符串
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
    获取chain的backbone原子Coordinates
    
    Args:
        chain: PDBchain对象
    
    Returns:
        backbone原子Coordinates数组 (N, 3, 3)，其中N是residue数，3是N、CA、C原子，3是Coordinates
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
                # 如果原子不Exists，Skip
                continue
    
    return np.array(coords)


def build_mpnn_pt_file(pdb_file, interface_dir, output_dir):
    """
    BuildMPNN .ptfile
    
    Args:
        pdb_file: PDBfilePath
        interface_dir: 界面Maskdirectory
        output_dir: outputdirectory
    """
    # 解析PDB结构
    structure = parse_pdb_structure(pdb_file)
    if structure is None:
        return
    
    # 获取PDB ID
    pdb_id = os.path.basename(pdb_file).split('_')[0]
    
    # Processing每条chain
    for chain in structure.get_chains():
        chain_id = chain.id
        
        # 获取sequence
        sequence = get_chain_sequence(chain)
        if not sequence:
            continue
        
        # 获取backboneCoordinates
        backbone_coords = get_backbone_coords(chain)
        if backbone_coords.shape[0] == 0:
            continue
        
        # load界面Mask
        interface_mask_file = os.path.join(interface_dir, f"{pdb_id}_{chain_id}.npy")
        if os.path.exists(interface_mask_file):
            interface_mask = np.load(interface_mask_file)
            # 确保Mask长度与sequence长度匹配
            if len(interface_mask) != len(sequence):
                # 如果不匹配，Create长度为sequence长度的全FalseMask
                interface_mask = np.zeros(len(sequence), dtype=bool)
        else:
            # 如果没有界面Maskfile，Create全FalseMask
            interface_mask = np.zeros(len(sequence), dtype=bool)
        
        # CreateMPNNData字典
        data = {
            'sequence': sequence,
            'backbone_coords': torch.tensor(backbone_coords, dtype=torch.float32),
            'interface_mask': torch.tensor(interface_mask, dtype=torch.bool)
        }
        
        # save.ptfile
        output_file = os.path.join(output_dir, f"{pdb_id}_{chain_id}.pt")
        torch.save(data, output_file)
        print(f"已save {output_file}")


def main():
    """
    主Function
    """
    parser = argparse.ArgumentParser(description='将PDBfile转换为ProteinMPNN可直接Train的.ptfile')
    parser.add_argument('--input_dir', required=True, help='inputPDBfiledirectory')
    parser.add_argument('--interface_dir', default='data/processed/interface_masks', help='界面Maskdirectory')
    parser.add_argument('--output_dir', default='data/processed/mpnn_pt', help='outputdirectory')
    
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
            build_mpnn_pt_file(pdb_file, args.interface_dir, args.output_dir)
        except Exception as e:
            print(f"Processing {pdb_file} 时出错：{str(e)}")
    
    print("MPNN .ptfileBuildComplete")


if __name__ == "__main__":
    main()
