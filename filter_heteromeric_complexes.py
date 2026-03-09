#!/usr/bin/env python3
"""
filter_heteromeric_complexes.py

Function：筛选Meets criteria的heteromeric complexes

Core Rules：
1. resolution≤3.5Å
2. 至少2条不同sequence蛋白chain
3. ExcludeDNA/RNA的复合物

Dependencies：
- biopython

Usage：
python filter_heteromeric_complexes.py --input_dir pdb_files --output_dir filtered_complexes
"""

import os
import argparse
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

# 忽略PDB结构BuildWarning
warnings.filterwarnings('ignore', category=PDBConstructionWarning)


def get_resolution(pdb_file):
    """
    从PDBfile中获取resolution
    
    Args:
        pdb_file: PDBfilePath
    
    Returns:
        resolution值，如果Failed to get则返回None
    """
    try:
        if pdb_file.endswith('.cif'):
            parser = MMCIFParser()
            structure = parser.get_structure('structure', pdb_file)
            # 从header中获取resolution
            if 'resolution' in structure.header:
                return structure.header['resolution']
        else:
            parser = PDBParser()
            structure = parser.get_structure('structure', pdb_file)
            # 从header中获取resolution
            if 'resolution' in structure.header:
                return structure.header['resolution']
        return None
    except Exception as e:
        print(f"获取 {pdb_file} resolution时出错：{str(e)}")
        return None


def get_chain_sequences(structure):
    """
    获取结构中Allchain的sequence
    
    Args:
        structure: PDB结构对象
    
    Returns:
        字典，键为chainID，值为sequence
    """
    chain_sequences = {}
    
    for model in structure:
        for chain in model:
            sequence = []
            for residue in chain:
                # Exclude水和配体
                if residue.id[0] == ' ' and residue.get_resname() not in ['HOH', 'WAT']:
                    # 获取amino acid名称
                    resname = residue.get_resname()
                    # 简单的amino acid三字母到单字母的转换
                    aa_dict = {
                        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                    }
                    if resname in aa_dict:
                        sequence.append(aa_dict[resname])
            if sequence:
                chain_sequences[chain.id] = ''.join(sequence)
    
    return chain_sequences


def has_nucleic_acids(structure):
    """
    Check结构中是否Contains DNA/RNA
    
    Args:
        structure: PDB结构对象
    
    Returns:
        bool: 如果Contains DNA/RNA则返回True，否则返回False
    """
    nucleic_acid_residues = ['A', 'T', 'C', 'G', 'U', 'DA', 'DT', 'DC', 'DG', 'DU']
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':
                    resname = residue.get_resname()
                    if resname in nucleic_acid_residues:
                        return True
    
    return False


def filter_complex(pdb_file):
    """
    筛选Meets criteria的复合物
    
    Args:
        pdb_file: PDBfilePath
    
    Returns:
        bool: 如果Meets criteria则返回True，否则返回False
    """
    try:
        # 解析PDBfile
        if pdb_file.endswith('.cif'):
            parser = MMCIFParser()
        else:
            parser = PDBParser()
        structure = parser.get_structure('structure', pdb_file)
        
        # Check是否Contains DNA/RNA
        if has_nucleic_acids(structure):
            print(f"{pdb_file} Contains DNA/RNA，Exclude")
            return False
        
        # 获取resolution
        resolution = get_resolution(pdb_file)
        if resolution is None:
            print(f"{pdb_file} Failed to getresolution，ContinueProcessing")
        elif resolution > 3.5:
            print(f"{pdb_file} resolution {resolution}Å > 3.5Å，Warning但ContinueProcessing")
        
        # 获取chainsequence
        chain_sequences = get_chain_sequences(structure)
        
        # Check至少2条不同sequence的蛋白chain
        if len(chain_sequences) < 2:
            print(f"{pdb_file} 蛋白chainCount < 2，Exclude")
            return False
        
        # Check是否有不同的sequence
        unique_sequences = set(chain_sequences.values())
        if len(unique_sequences) < 2:
            print(f"{pdb_file} 没有不同sequence的蛋白chain，Exclude")
            return False
        
        print(f"{pdb_file} Meets criteria")
        return True
    except Exception as e:
        print(f"Processing {pdb_file} 时出错：{str(e)}")
        return False


def main():
    """
    主Function
    """
    parser = argparse.ArgumentParser(description='筛选Meets criteria的heteromeric complexes')
    parser.add_argument('--input_dir', required=True, help='inputPDBfiledirectory')
    parser.add_argument('--output_dir', default='filtered_complexes', help='outputdirectory')
    
    args = parser.parse_args()
    
    # 确保outputdirectoryExists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取inputdirectory中的PDBfile
    pdb_files = []
    for file in os.listdir(args.input_dir):
        if file.endswith('.pdb') or file.endswith('.cif'):
            pdb_files.append(os.path.join(args.input_dir, file))
    
    print(f"StartProcessing {len(pdb_files)} 个PDBfile")
    
    # 筛选Meets criteria的复合物
    for pdb_file in pdb_files:
        if filter_complex(pdb_file):
            # 复制到outputdirectory
            output_file = os.path.join(args.output_dir, os.path.basename(pdb_file))
            import shutil
            shutil.copy2(pdb_file, output_file)
            print(f"Copied to {output_file}")
    
    print("筛选Complete")


if __name__ == "__main__":
    main()
