.PHONY: all preprocess train evaluate clean help

# 默认目标：运行完整流程
all: preprocess train evaluate
	@echo "✅ 全流程完成！"

# 帮助信息
help:
	@echo "ComplexMPNN - 蛋白质复合物序列设计模型"
	@echo ""
	@echo "使用方法："
	@echo "  make all          - 运行完整流程（预处理→训练→评估）"
	@echo "  make preprocess   - 仅运行预处理阶段"
	@echo "  make train        - 仅运行训练阶段"
	@echo "  make evaluate     - 仅运行评估阶段"
	@echo "  make clean        - 清理生成的文件"
	@echo "  make help         - 显示帮助信息"
	@echo ""
	@echo "注意：请确保已激活complexmpnn conda环境"

# 预处理阶段
preprocess:
	@echo "=========================================="
	@echo "阶段1: 数据预处理"
	@echo "=========================================="
	@if [ ! -f "data/processed/mpnn_pt/1cg5_A.pt" ]; then \
		echo "运行预处理流程..."; \
		bash run_preprocess_test.sh 2>&1 | tee preprocess.log; \
		if [ $$? -ne 0 ]; then \
			echo "❌ 预处理失败，请检查 preprocess.log"; \
			exit 1; \
		fi; \
	else \
		echo "✅ 预处理数据已存在，跳过"; \
	fi
	@echo ""

# 训练阶段
train: preprocess
	@echo "=========================================="
	@echo "阶段2: 模型训练"
	@echo "=========================================="
	@if [ ! -f "checkpoints/best_complexmpnn.pt" ]; then \
		echo "运行训练流程..."; \
		bash run_train_test.sh 2>&1 | tee train.log; \
		if [ $$? -ne 0 ]; then \
			echo "❌ 训练失败，请检查 train.log"; \
			exit 1; \
		fi; \
	else \
		echo "✅ 模型checkpoint已存在，跳过"; \
	fi
	@echo ""

# 评估阶段
evaluate: train
	@echo "=========================================="
	@echo "阶段3: 模型评估"
	@echo "=========================================="
	@if [ ! -f "logs/evaluation/combined_evaluation_results.csv" ]; then \
		echo "运行评估流程..."; \
		bash run_evaluation_test.sh 2>&1 | tee evaluate.log; \
		if [ $$? -ne 0 ]; then \
			echo "❌ 评估失败，请检查 evaluate.log"; \
			exit 1; \
		fi; \
	else \
		echo "✅ 评估结果已存在，跳过"; \
	fi
	@echo ""
	@echo "=========================================="
	@echo "📊 评估结果摘要"
	@echo "=========================================="
	@if [ -f "logs/evaluation/combined_evaluation_results.csv" ]; then \
		cat logs/evaluation/combined_evaluation_results.csv; \
	fi
	@echo ""

# 清理生成的文件
clean:
	@echo "清理生成的文件..."
	@rm -f preprocess.log train.log evaluate.log evaluation_test.log
	@rm -rf checkpoints/*.pt checkpoints/*.csv
	@rm -rf logs/*.log logs/evaluation/*
	@echo "✅ 清理完成！"
	@echo "注意：data/ 目录下的预处理数据未删除，如需完全清理请手动删除"
