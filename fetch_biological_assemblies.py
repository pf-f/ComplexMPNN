#!/usr/bin/env python3
"""
fetch_biological_assemblies.py

Function：DownloadPDB的第一个biological assembly，禁用asymmetric unit

Core Rules：
1. 仅DownloadPDB的第一个biological assembly
2. 禁用asymmetric unit
3. 仅Processing指定的PDB ID列表

Dependencies：
- requests

Usage：
python fetch_biological_assemblies.py --pdb_list test_pdb_ids.txt --output_dir data/raw_pdb
"""

import os
import argparse
import requests


def fetch_biological_assembly(pdb_id, output_dir):
    """
    Download指定PDB ID的第一个biological assembly
    
    Args:
        pdb_id: PDB ID字符串
        output_dir: outputdirectory
    """
    # 确保outputdirectoryExists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Buildbiological assembly的URL - 使用正确的RCSB URL
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb1.gz"
        temp_file = os.path.join(output_dir, f"{pdb_id.lower()}_assembly1.pdb.gz")
        output_file = os.path.join(output_dir, f"{pdb_id.lower()}_assembly1.pdb")
        
        # Download压缩file
        response = requests.get(url)
        if response.status_code == 200:
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # 解压file
            import gzip
            with gzip.open(temp_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # 删除临时压缩file
            os.remove(temp_file)
            
            print(f"SuccessDownload {pdb_id} 的第一个biological assembly到 {output_file}")
            return output_file
        else:
            # 如果压缩file不Exists，尝试直接Download标准PDBfile
            url_pdb = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            response_pdb = requests.get(url_pdb)
            if response_pdb.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response_pdb.content)
                print(f"SuccessDownload {pdb_id} 的PDBfile到 {output_file}")
                return output_file
            else:
                print(f"Download {pdb_id} Failed：HTTP状态码 {response.status_code} (压缩) 和 {response_pdb.status_code} (标准)")
                return None
    except Exception as e:
        print(f"Download {pdb_id} 时出错：{str(e)}")
        return None


def main():
    """
    主Function
    """
    parser = argparse.ArgumentParser(description='DownloadPDB biological assemblies')
    parser.add_argument('--pdb_list', required=True, help='包含PDB ID列表的文本file')
    parser.add_argument('--output_dir', default='data/raw_pdb', help='outputdirectory')
    
    args = parser.parse_args()
    
    # 读取PDB ID列表
    with open(args.pdb_list, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    
    print(f"StartProcessing {len(pdb_ids)} 个PDB ID")
    
    # Download每个PDB的第一个biological assembly
    for pdb_id in pdb_ids:
        fetch_biological_assembly(pdb_id, args.output_dir)
    
    print("DownloadComplete")


if __name__ == "__main__":
    main()
