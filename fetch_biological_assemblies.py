#!/usr/bin/env python3
"""
fetch_biological_assemblies.py

功能：下载PDB的第一个biological assembly，禁用asymmetric unit

核心规则：
1. 仅下载PDB的第一个biological assembly
2. 禁用asymmetric unit
3. 仅处理指定的PDB ID列表

依赖：
- requests

使用方法：
python fetch_biological_assemblies.py --pdb_list test_pdb_ids.txt --output_dir data/raw_pdb
"""

import os
import argparse
import requests


def fetch_biological_assembly(pdb_id, output_dir):
    """
    下载指定PDB ID的第一个biological assembly
    
    Args:
        pdb_id: PDB ID字符串
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 构建biological assembly的URL - 使用正确的RCSB URL
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb1.gz"
        temp_file = os.path.join(output_dir, f"{pdb_id.lower()}_assembly1.pdb.gz")
        output_file = os.path.join(output_dir, f"{pdb_id.lower()}_assembly1.pdb")
        
        # 下载压缩文件
        response = requests.get(url)
        if response.status_code == 200:
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # 解压文件
            import gzip
            with gzip.open(temp_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # 删除临时压缩文件
            os.remove(temp_file)
            
            print(f"成功下载 {pdb_id} 的第一个biological assembly到 {output_file}")
            return output_file
        else:
            # 如果压缩文件不存在，尝试直接下载标准PDB文件
            url_pdb = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            response_pdb = requests.get(url_pdb)
            if response_pdb.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response_pdb.content)
                print(f"成功下载 {pdb_id} 的PDB文件到 {output_file}")
                return output_file
            else:
                print(f"下载 {pdb_id} 失败：HTTP状态码 {response.status_code} (压缩) 和 {response_pdb.status_code} (标准)")
                return None
    except Exception as e:
        print(f"下载 {pdb_id} 时出错：{str(e)}")
        return None


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='下载PDB biological assemblies')
    parser.add_argument('--pdb_list', required=True, help='包含PDB ID列表的文本文件')
    parser.add_argument('--output_dir', default='data/raw_pdb', help='输出目录')
    
    args = parser.parse_args()
    
    # 读取PDB ID列表
    with open(args.pdb_list, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    
    print(f"开始处理 {len(pdb_ids)} 个PDB ID")
    
    # 下载每个PDB的第一个biological assembly
    for pdb_id in pdb_ids:
        fetch_biological_assembly(pdb_id, args.output_dir)
    
    print("下载完成")


if __name__ == "__main__":
    main()
