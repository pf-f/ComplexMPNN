#!/bin/bash
#
# run_evaluation_test.sh
#
# 功能：一键运行评估测试，包括验证、序列恢复计算、AF-Multimer预测、结果分析
#
# 使用方法：
# bash run_evaluation_test.sh
#

set -e

# 日志文件
LOG_FILE="evaluation_test.log"
EXEC_DIR=$(pwd)

# 时间戳函数
log_timestamp() {
    date +"[%Y-%m-%d %H:%M:%S]"
}

# 错误处理函数
handle_error() {
    echo "$(log_timestamp) 错误: $1" | tee -a "$LOG_FILE"
    exit 1
}

# 初始化日志
echo "$(log_timestamp) 开始评估测试..." > "$LOG_FILE"
echo "$(log_timestamp) 工作目录: $EXEC_DIR" | tee -a "$LOG_FILE"

# 步骤1: 检查Python环境
echo "$(log_timestamp) 检查Python环境..." | tee -a "$LOG_FILE"
if ! command -v python3 &> /dev/null; then
    handle_error "Python3未找到"
fi
python3 --version | tee -a "$LOG_FILE"

# 步骤2: 检查依赖
echo "$(log_timestamp) 检查依赖..." | tee -a "$LOG_FILE"
python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null || handle_error "torch未安装"
python3 -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null || handle_error "numpy未安装"
python3 -c "import pandas; print('pandas', pandas.__version__)" 2>/dev/null || handle_error "pandas未安装"
python3 -c "import matplotlib; print('matplotlib', matplotlib.__version__)" 2>/dev/null || handle_error "matplotlib未安装"

# 步骤3: 创建必要的目录
echo "$(log_timestamp) 创建必要的目录..." | tee -a "$LOG_FILE"
mkdir -p logs/evaluation
mkdir -p logs/evaluation/af_output

# 步骤4: 验证评估模块
echo "$(log_timestamp) 步骤1: 验证评估模块..." | tee -a "$LOG_FILE"
if python3 check_evaluation.py 2>&1 | tee -a "$LOG_FILE"; then
    echo "$(log_timestamp) 评估模块验证完成" | tee -a "$LOG_FILE"
else
    handle_error "评估模块验证失败"
fi

# 步骤5: 计算序列恢复指标
echo "$(log_timestamp) 步骤2: 计算序列恢复指标..." | tee -a "$LOG_FILE"
if [ -f "checkpoints/best_complexmpnn.pt" ]; then
    if python3 interface_recovery.py --ckpt checkpoints/best_complexmpnn.pt --test_split test.txt 2>&1 | tee -a "$LOG_FILE"; then
        echo "$(log_timestamp) 序列恢复指标计算完成" | tee -a "$LOG_FILE"
    else
        handle_error "序列恢复指标计算失败"
    fi
else
    echo "$(log_timestamp) 警告: 模型checkpoint不存在，跳过序列恢复计算" | tee -a "$LOG_FILE"
fi

# 步骤6: 运行AlphaFold-Multimer（简化测试）
echo "$(log_timestamp) 步骤3: 运行AlphaFold-Multimer测试..." | tee -a "$LOG_FILE"
if python3 run_af_multimer.py --sequences "ACDEFGHIKLMNPQRSTVWY;YWVTSRQPNMLKIHGFEDCA" --output_dir logs/evaluation/af_output 2>&1 | tee -a "$LOG_FILE"; then
    echo "$(log_timestamp) AlphaFold-Multimer测试完成" | tee -a "$LOG_FILE"
else
    handle_error "AlphaFold-Multimer测试失败"
fi

# 步骤7: 分析结果
echo "$(log_timestamp) 步骤4: 分析评估结果..." | tee -a "$LOG_FILE"
if python3 analyze_results.py 2>&1 | tee -a "$LOG_FILE"; then
    echo "$(log_timestamp) 结果分析完成" | tee -a "$LOG_FILE"
else
    handle_error "结果分析失败"
fi

# 步骤8: 打印评估结果
echo "$(log_timestamp) 评估测试完成！" | tee -a "$LOG_FILE"
echo "$(log_timestamp) 日志文件: $LOG_FILE" | tee -a "$LOG_FILE"

echo -e "\n$(log_timestamp) 评估结果信息:" | tee -a "$LOG_FILE"
echo "$(log_timestamp) 评估结果目录: logs/evaluation" | tee -a "$LOG_FILE"
echo "$(log_timestamp) 生成的文件:" | tee -a "$LOG_FILE"
if [ -d "logs/evaluation" ]; then
    ls -lh logs/evaluation/ 2>/dev/null | tee -a "$LOG_FILE" || true
fi

echo -e "\n$(log_timestamp) 🎉 评估测试成功完成！" | tee -a "$LOG_FILE"
