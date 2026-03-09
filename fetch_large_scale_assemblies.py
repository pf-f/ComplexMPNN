#!/usr/bin/env python3
"""
大规模PDB biological assembliesDownload脚本
优化：支持多线程、Error重试、进度显示
"""

import os
import argparse
import requests
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time


def fetch_biological_assembly(pdb_id, output_dir, max_retries=3):
    """
    Download指定PDB ID的第一个biological assembly，支持重试
    
    Args:
        pdb_id: PDB ID字符串
        output_dir: outputdirectory
        max_retries: 最大重试次数
    
    Returns:
        (pdb_id, success, output_file或Error信息)
    """
    pdb_id = pdb_id.upper()
    output_file = os.path.join(output_dir, f"{pdb_id.lower()}_assembly1.pdb")
    
    # 如果file已存在，Skip
    if os.path.exists(output_file):
        return (pdb_id, True, output_file)
    
    for attempt in range(max_retries):
        try:
            # 首先尝试biological assembly
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb1.gz"
            temp_file = os.path.join(output_dir, f"{pdb_id.lower()}_assembly1.pdb.gz")
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                
                # 解压
                with gzip.open(temp_file, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                os.remove(temp_file)
                return (pdb_id, True, output_file)
            
            # 如果Failed，尝试标准PDB
            url_pdb = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response_pdb = requests.get(url_pdb, timeout=30)
            if response_pdb.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response_pdb.content)
                return (pdb_id, True, output_file)
            
            # 等待后重试
            if attempt < max_retries - 1:
                time.sleep(1)
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return (pdb_id, False, str(e))
    
    return (pdb_id, False, "Max retries exceeded")


def main():
    parser = argparse.ArgumentParser(description='大规模DownloadPDB biological assemblies')
    parser.add_argument('--pdb_list', required=True, help='包含PDB ID列表的文本file')
    parser.add_argument('--output_dir', default='data/raw_pdb', help='outputdirectory')
    parser.add_argument('--max_workers', type=int, default=10, help='并发线程数')
    parser.add_argument('--limit', type=int, default=None, help='限制Download数量（用于测试）')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取PDB ID列表
    with open(args.pdb_list, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    
    if args.limit:
        pdb_ids = pdb_ids[:args.limit]
    
    print(f"准备Download {len(pdb_ids)} 个PDBfile")
    print(f"使用 {args.max_workers} 个并发线程")
    
    # 统计已存在的file
    existing = 0
    for pdb_id in pdb_ids:
        output_file = os.path.join(args.output_dir, f"{pdb_id.lower()}_assembly1.pdb")
        if os.path.exists(output_file):
            existing += 1
    
    print(f"已存在 {existing} 个file，需要Download {len(pdb_ids) - existing} 个")
    
    # 多线程Download
    success_count = 0
    fail_count = 0
    failed_pdbs = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(fetch_biological_assembly, pdb_id, args.output_dir): pdb_id 
                  for pdb_id in pdb_ids}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Download进度"):
            pdb_id, success, result = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_pdbs.append((pdb_id, result))
    
    print(f"\nDownloadComplete！")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    
    if failed_pdbs:
        print(f"\nFailed的PDB:")
        for pdb_id, error in failed_pdbs[:20]:
            print(f"  {pdb_id}: {error}")
        if len(failed_pdbs) > 20:
            print(f"  ... 还有 {len(failed_pdbs) - 20} 个")


if __name__ == "__main__":
    main()
