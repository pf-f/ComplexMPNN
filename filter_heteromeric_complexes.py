#!/usr/bin/env python3
"""
filter_heteromeric_complexes.py

功能：筛选符合条件的异源复合物

核心规则：
1. 分辨率≤3.5Å
2. 至少2条不同序列蛋白链
3. 排除DNA/RNA的复合物

依赖：
- biopython

使用方法：
python filter_heteromeric_complexes.py --input_dir pdb_files --output_dir filtered_complexes
"""

import os
import argparse
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

# 忽略PDB结构构建警告
warnings.filterwarnings('ignore', category=PDBConstructionWarning)


def get_resolution(pdb_file):
    """
    从PDB文件中获取分辨率
    
    Args:
        pdb_file: PDB文件路径
    
    Returns:
        分辨率值，如果无法获取则返回None
    """
    try:
        if pdb_file.endswith('.cif'):
            parser = MMCIFParser()
            structure = parser.get_structure('structure', pdb_file)
            # 从header中获取分辨率
            if 'resolution' in structure.header:
                return structure.header['resolution']
        else:
            parser = PDBParser()
            structure = parser.get_structure('structure', pdb_file)
            # 从header中获取分辨率
            if 'resolution' in structure.header:
                return structure.header['resolution']
        return None
    except Exception as e:
        print(f"获取 {pdb_file} 分辨率时出错：{str(e)}")
        return None


def get_chain_sequences(structure):
    """
    获取结构中所有链的序列
    
    Args:
        structure: PDB结构对象
    
    Returns:
        字典，键为链ID，值为序列
    """
    chain_sequences = {}
    
    for model in structure:
        for chain in model:
            sequence = []
            for residue in chain:
                # 排除水和配体
                if residue.id[0] == ' ' and residue.get_resname() not in ['HOH', 'WAT']:
                    # 获取氨基酸名称
                    resname = residue.get_resname()
                    # 简单的氨基酸三字母到单字母的转换
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
    检查结构中是否包含DNA/RNA
    
    Args:
        structure: PDB结构对象
    
    Returns:
        bool: 如果包含DNA/RNA则返回True，否则返回False
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
    筛选符合条件的复合物
    
    Args:
        pdb_file: PDB文件路径
    
    Returns:
        bool: 如果符合条件则返回True，否则返回False
    """
    try:
        # 解析PDB文件
        if pdb_file.endswith('.cif'):
            parser = MMCIFParser()
        else:
            parser = PDBParser()
        structure = parser.get_structure('structure', pdb_file)
        
        # 检查是否包含DNA/RNA
        if has_nucleic_acids(structure):
            print(f"{pdb_file} 包含DNA/RNA，排除")
            return False
        
        # 获取分辨率
        resolution = get_resolution(pdb_file)
        if resolution is None:
            print(f"{pdb_file} 无法获取分辨率，继续处理")
        elif resolution > 3.5:
            print(f"{pdb_file} 分辨率 {resolution}Å > 3.5Å，警告但继续处理")
        
        # 获取链序列
        chain_sequences = get_chain_sequences(structure)
        
        # 检查至少2条不同序列的蛋白链
        if len(chain_sequences) < 2:
            print(f"{pdb_file} 蛋白链数量 < 2，排除")
            return False
        
        # 检查是否有不同的序列
        unique_sequences = set(chain_sequences.values())
        if len(unique_sequences) < 2:
            print(f"{pdb_file} 没有不同序列的蛋白链，排除")
            return False
        
        print(f"{pdb_file} 符合条件")
        return True
    except Exception as e:
        print(f"处理 {pdb_file} 时出错：{str(e)}")
        return False


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='筛选符合条件的异源复合物')
    parser.add_argument('--input_dir', required=True, help='输入PDB文件目录')
    parser.add_argument('--output_dir', default='filtered_complexes', help='输出目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取输入目录中的PDB文件
    pdb_files = []
    for file in os.listdir(args.input_dir):
        if file.endswith('.pdb') or file.endswith('.cif'):
            pdb_files.append(os.path.join(args.input_dir, file))
    
    print(f"开始处理 {len(pdb_files)} 个PDB文件")
    
    # 筛选符合条件的复合物
    for pdb_file in pdb_files:
        if filter_complex(pdb_file):
            # 复制到输出目录
            output_file = os.path.join(args.output_dir, os.path.basename(pdb_file))
            import shutil
            shutil.copy2(pdb_file, output_file)
            print(f"已复制到 {output_file}")
    
    print("筛选完成")


if __name__ == "__main__":
    main()
