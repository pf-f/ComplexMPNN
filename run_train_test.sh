#!/bin/bash

# run_train_test.sh
# 一键运行训练测试脚本，包含日志输出、错误处理、训练过程关键指标打印

# 设置错误时退出
set -e

# 设置日志文件
LOG_FILE="train_test.log"

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
pip list | grep -E "torch|pyyaml|numpy" 2>&1 | tee -a "$LOG_FILE"

# 创建必要的目录
log "创建必要的目录..."
mkdir -p checkpoints logs

# 步骤1: 验证训练模块
log "步骤1: 验证训练模块..."
if python check_train.py; then
    log "训练模块验证完成"
else
    error_exit "训练模块验证失败"
fi

# 步骤2: 运行训练
log "步骤2: 运行训练..."
if python train_complex_mpnn.py --config config.yaml; then
    log "训练完成"
else
    error_exit "训练失败"
fi

# 总结
log "训练测试完成！"
log "日志文件: $LOG_FILE"

# 显示checkpoint信息
log "\nCheckpoint信息:"
if [ -d "checkpoints" ]; then
    log "Checkpoint目录: checkpoints"
    log "Checkpoint文件:"
    ls -lh checkpoints/ 2>/dev/null | tee -a "$LOG_FILE"
else
    log "警告: checkpoints目录不存在"
fi

# 显示训练日志信息
if [ -f "logs/train.log" ]; then
    log "\n训练日志信息:"
    log "训练日志文件: logs/train.log"
    log "训练日志最后10行:"
    tail -n 10 logs/train.log 2>/dev/null | tee -a "$LOG_FILE"
fi
