#!/bin/bash
#
# run_all.sh - ComplexMPNN 全流程一键运行脚本
#
# 功能：按顺序执行「预处理→训练→评估」全流程
# 支持：错误处理、日志记录、流程中断恢复
#
# 使用方法：
#   bash run_all.sh              # 运行完整流程
#   bash run_all.sh preprocess   # 仅运行预处理
#   bash run_all.sh train        # 仅运行训练
#   bash run_all.sh evaluate     # 仅运行评估
#

set -e

# 配置
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/full_pipeline.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}

# 错误处理函数
handle_error() {
    log_error "流程在第 $1 阶段失败！"
    log_error "请查看日志文件: $LOG_FILE"
    exit 1
}

# 检查conda环境
check_conda_env() {
    log_info "检查conda环境..."
    if [[ "$CONDA_DEFAULT_ENV" != "complexmpnn" ]]; then
        log_warning "当前环境不是 complexmpnn，建议激活: conda activate complexmpnn"
        read -p "是否继续运行? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "用户取消运行"
            exit 0
        fi
    else
        log_success "conda环境检查通过: complexmpnn"
    fi
}

# 预处理阶段
run_preprocess() {
    log_info "=========================================="
    log_info "阶段1: 数据预处理"
    log_info "=========================================="
    
    # 检查是否已完成
    if [ -f "data/processed/mpnn_pt/1cg5_A.pt" ]; then
        log_success "预处理数据已存在，跳过"
        return 0
    fi
    
    log_info "运行预处理流程..."
    if bash run_preprocess_test.sh 2>&1 | tee -a "$LOG_FILE"; then
        log_success "预处理完成"
    else
        handle_error "预处理"
    fi
}

# 训练阶段
run_train() {
    log_info "=========================================="
    log_info "阶段2: 模型训练"
    log_info "=========================================="
    
    # 确保预处理已完成
    if [ ! -f "data/processed/mpnn_pt/1cg5_A.pt" ]; then
        log_error "预处理数据不存在，请先运行预处理阶段"
        handle_error "训练（前置条件缺失）"
    fi
    
    # 检查是否已完成
    if [ -f "checkpoints/best_complexmpnn.pt" ]; then
        log_success "模型checkpoint已存在，跳过"
        return 0
    fi
    
    log_info "运行训练流程..."
    if bash run_train_test.sh 2>&1 | tee -a "$LOG_FILE"; then
        log_success "训练完成"
    else
        handle_error "训练"
    fi
}

# 评估阶段
run_evaluate() {
    log_info "=========================================="
    log_info "阶段3: 模型评估"
    log_info "=========================================="
    
    # 确保训练已完成
    if [ ! -f "checkpoints/best_complexmpnn.pt" ]; then
        log_error "模型checkpoint不存在，请先运行训练阶段"
        handle_error "评估（前置条件缺失）"
    fi
    
    # 检查是否已完成
    if [ -f "logs/evaluation/combined_evaluation_results.csv" ]; then
        log_success "评估结果已存在，跳过"
        return 0
    fi
    
    log_info "运行评估流程..."
    if bash run_evaluation_test.sh 2>&1 | tee -a "$LOG_FILE"; then
        log_success "评估完成"
    else
        handle_error "评估"
    fi
}

# 显示结果摘要
show_results() {
    log_info "=========================================="
    log_info "📊 评估结果摘要"
    log_info "=========================================="
    
    if [ -f "logs/evaluation/combined_evaluation_results.csv" ]; then
        echo ""
        cat "logs/evaluation/combined_evaluation_results.csv"
        echo ""
    else
        log_warning "未找到评估结果文件"
    fi
    
    log_info "=========================================="
    log_info "生成的文件位置："
    log_info "  - 模型checkpoint: checkpoints/"
    log_info "  - 评估结果: logs/evaluation/"
    log_info "  - 完整日志: $LOG_FILE"
    log_info "=========================================="
}

# 显示帮助信息
show_help() {
    echo "ComplexMPNN - 蛋白质复合物序列设计模型"
    echo ""
    echo "使用方法："
    echo "  bash run_all.sh              运行完整流程（预处理→训练→评估）"
    echo "  bash run_all.sh preprocess   仅运行预处理阶段"
    echo "  bash run_all.sh train        仅运行训练阶段"
    echo "  bash run_all.sh evaluate     仅运行评估阶段"
    echo "  bash run_all.sh help         显示帮助信息"
    echo ""
    echo "注意事项："
    echo "  1. 请确保已激活 complexmpnn conda 环境"
    echo "  2. 流程支持中断恢复，已完成的阶段会自动跳过"
    echo "  3. 完整日志保存在: $LOG_FILE"
}

# 主函数
main() {
    # 创建日志目录
    mkdir -p "$LOG_DIR"
    
    # 初始化日志
    echo "==========================================" > "$LOG_FILE"
    echo "ComplexMPNN 全流程运行日志" >> "$LOG_FILE"
    echo "开始时间: $(date)" >> "$LOG_FILE"
    echo "==========================================" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    log_info "开始运行 ComplexMPNN 全流程"
    
    # 检查conda环境
    check_conda_env
    
    # 根据参数执行
    case "${1:-all}" in
        preprocess)
            run_preprocess
            ;;
        train)
            run_train
            ;;
        evaluate)
            run_evaluate
            ;;
        help)
            show_help
            ;;
        all)
            run_preprocess
            run_train
            run_evaluate
            show_results
            log_success "🎉 全流程完成！"
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
