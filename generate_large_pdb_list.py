#!/usr/bin/env python3
"""
生成大规模PDB ID列表
从1A00到1Z99，包含上千个PDB ID
"""

import string

def generate_pdb_ids():
    """生成从1A00到1Z99的PDB ID列表"""
    pdb_ids = []
    
    # 第一个字符：1
    first_char = '1'
    
    # 第二个字符：A-Z
    for second_char in string.ascii_uppercase:
        # 第三个字符：0-9，A-Z
        for third_char in string.digits + string.ascii_uppercase:
            # 第四个字符：0-9
            for fourth_char in string.digits:
                pdb_id = f"{first_char}{second_char}{third_char}{fourth_char}"
                pdb_ids.append(pdb_id)
    
    return pdb_ids

def main():
    print("生成大规模PDB ID列表...")
    pdb_ids = generate_pdb_ids()
    print(f"共生成 {len(pdb_ids)} 个PDB ID")
    
    # save到file
    output_file = "large_pdb_ids.txt"
    with open(output_file, 'w') as f:
        for pdb_id in pdb_ids:
            f.write(f"{pdb_id}\n")
    
    print(f"已save到 {output_file}")

if __name__ == "__main__":
    main()
