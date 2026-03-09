#!/bin/bash

# run_preprocess_test.sh
# 一键运行预处理测试脚本，包含错误处理和日志输出

# 设置错误时退出
set -e

# 设置日志文件
LOG_FILE="preprocess_test.log"

# 清空日志文件
> "$LOG_FILE"

# 定义日志函数
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 定义错误处理函数
error_exit() {
    log "错误: $1"
    exit 1
}

# 检查Python环境
log "检查Python环境..."
python --version 2>&1 | tee -a "$LOG_FILE"

# 检查依赖
log "检查依赖..."
pip list | grep -E "biopython|numpy|torch|scikit-learn|requests" 2>&1 | tee -a "$LOG_FILE"

# 检查test_pdb_ids.txt文件
if [ ! -f "test_pdb_ids.txt" ]; then
    log "创建test_pdb_ids.txt文件..."
    cat > test_pdb_ids.txt << EOF
1A00
1B00
1C00
1D00
1E00
1F00
1G00
1H00
1I00
1J00
EOF
    log "test_pdb_ids.txt文件创建完成"
fi

# 创建必要的目录
log "创建必要的目录..."
mkdir -p data/raw_pdb data/processed/structures data/processed/interface_masks data/processed/mpnn_pt data/splits

# 步骤1: 下载PDB数据
log "步骤1: 下载PDB数据..."
if python fetch_biological_assemblies.py --pdb_list test_pdb_ids.txt --output_dir data/raw_pdb; then
    log "PDB数据下载完成"
else
    error_exit "PDB数据下载失败"
fi

# 检查是否有下载的文件
PDB_COUNT=$(ls -1 data/raw_pdb 2>/dev/null | wc -l)
if [ "$PDB_COUNT" -eq 0 ]; then
    log "警告: 没有下载到任何PDB文件，可能是因为PDB ID不存在或网络问题"
    log "预处理测试完成！"
    log "日志文件: $LOG_FILE"
    exit 0
fi

# 步骤2: 筛选复合物
log "步骤2: 筛选复合物..."
if python filter_heteromeric_complexes.py --input_dir data/raw_pdb --output_dir data/processed/structures; then
    log "复合物筛选完成"
else
    error_exit "复合物筛选失败"
fi

# 检查是否有筛选后的文件
STRUCTURE_COUNT=$(ls -1 data/processed/structures 2>/dev/null | wc -l)
if [ "$STRUCTURE_COUNT" -eq 0 ]; then
    log "警告: 没有筛选到任何复合物，可能是因为所有PDB文件都不符合条件"
    log "预处理测试完成！"
    log "日志文件: $LOG_FILE"
    exit 0
fi

# 步骤3: 检测界面残基
log "步骤3: 检测界面残基..."
if python detect_interfaces.py --input_dir data/processed/structures --output_dir data/processed/interface_masks; then
    log "界面残基检测完成"
else
    error_exit "界面残基检测失败"
fi

# 步骤4: 构建MPNN .pt文件
log "步骤4: 构建MPNN .pt文件..."
if python build_mpnn_pt_files.py --input_dir data/processed/structures --interface_dir data/processed/interface_masks --output_dir data/processed/mpnn_pt; then
    log "MPNN .pt文件构建完成"
else
    error_exit "MPNN .pt文件构建失败"
fi

# 步骤5: 聚类和切分数据集
log "步骤5: 聚类和切分数据集..."
if python cluster_and_split.py --input_dir data/processed/mpnn_pt --output_dir data/splits; then
    log "数据集聚类和切分完成"
else
    error_exit "数据集聚类和切分失败"
fi

# 步骤6: 验证预处理结果
log "步骤6: 验证预处理结果..."
if python check_preprocess.py --mpnn_dir data/processed/mpnn_pt --split_dir data/splits; then
    log "预处理结果验证完成"
else
    error_exit "预处理结果验证失败"
fi

# 总结
log "预处理测试完成！"
log "日志文件: $LOG_FILE"

# 显示结果统计
log "\n结果统计:"
log "下载的PDB文件数: $(ls -l data/raw_pdb | wc -l)"
log "筛选后的复合物数: $(ls -l data/processed/structures | wc -l)"
log "生成的界面掩码数: $(ls -l data/processed/interface_masks | wc -l)"
log "生成的MPNN .pt文件数: $(ls -l data/processed/mpnn_pt | wc -l)"

# 显示数据集切分结果
if [ -f "data/splits/train.txt" ]; then
    log "训练集样本数: $(wc -l < data/splits/train.txt)"
fi
if [ -f "data/splits/val.txt" ]; then
    log "验证集样本数: $(wc -l < data/splits/val.txt)"
fi
if [ -f "data/splits/test.txt" ]; then
    log "测试集样本数: $(wc -l < data/splits/test.txt)"
fi
