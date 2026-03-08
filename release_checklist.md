# ComplexMPNN 开源发布检查清单

## 📋 项目文件检查

### 核心代码文件
- [ ] `loss_functions.py` - 损失函数定义
- [ ] `train_complex_mpnn.py` - 训练脚本
- [ ] `interface_recovery.py` - 序列恢复评估
- [ ] `run_af_multimer.py` - AlphaFold-Multimer对接
- [ ] `analyze_results.py` - 结果分析和可视化

### 预处理脚本
- [ ] `fetch_biological_assemblies.py` - PDB下载
- [ ] `filter_heteromeric_complexes.py` - 复合物筛选
- [ ] `detect_interfaces.py` - 界面检测
- [ ] `build_mpnn_pt_files.py` - MPNN数据构建
- [ ] `cluster_and_split.py` - 数据聚类和切分

### 自动化脚本
- [ ] `Makefile` - Make自动化
- [ ] `run_all.sh` - Shell全流程脚本
- [ ] `run_preprocess_test.sh` - 预处理测试
- [ ] `run_train_test.sh` - 训练测试
- [ ] `run_evaluation_test.sh` - 评估测试

### 验证脚本
- [ ] `check_preprocess.py` - 预处理验证
- [ ] `check_train.py` - 训练验证
- [ ] `check_evaluation.py` - 评估验证
- [ ] `check_full_pipeline.py` - 全流程验证

### 配置和文档
- [ ] `config.yaml` - 配置文件
- [ ] `environment.yml` - Conda环境配置
- [ ] `.gitignore` - Git忽略文件
- [ ] `README.md` - 项目说明文档
- [ ] `ComplexMPNN.md` - 原始任务文档

### 示例文件
- [ ] `examples/fixed_chain_design_example.py` - Fixed-chain模式示例
- [ ] `examples/joint_design_example.py` - Joint-design模式示例
- [ ] `examples/interface_recovery_example.py` - 序列恢复示例

### 测试文件
- [ ] `test_pdb_ids.txt` - 测试PDB ID列表

## 🧪 功能验证检查

### 环境配置
- [ ] Conda环境可以通过 `environment.yml` 创建
- [ ] 所有Python依赖已正确安装
- [ ] `conda activate complexmpnn` 可以正常激活环境

### 预处理流程
- [ ] `python check_preprocess.py` 运行成功
- [ ] `bash run_preprocess_test.sh` 运行成功
- [ ] `data/processed/mpnn_pt/` 目录包含 .pt 文件
- [ ] `data/processed/interface_masks/` 目录包含界面掩码
- [ ] `data/splits/` 目录包含 train/val/test.txt

### 训练流程
- [ ] `python check_train.py` 运行成功
- [ ] `bash run_train_test.sh` 运行成功
- [ ] `checkpoints/best_complexmpnn.pt` 文件存在且可加载
- [ ] 训练过程loss正常下降
- [ ] 验证集loss正常计算

### 评估流程
- [ ] `python check_evaluation.py` 运行成功
- [ ] `bash run_evaluation_test.sh` 运行成功
- [ ] `logs/evaluation/sequence_recovery_results.pt` 存在
- [ ] `logs/evaluation/combined_evaluation_results.csv` 存在
- [ ] `logs/evaluation/*.png` 可视化图表存在

### 全流程自动化
- [ ] `make help` 显示正确帮助信息
- [ ] `make all` 可以完整运行全流程
- [ ] `bash run_all.sh` 可以完整运行全流程
- [ ] 流程支持中断恢复（已完成的阶段自动跳过）
- [ ] `python check_full_pipeline.py` 验证通过

### 示例代码
- [ ] `python examples/fixed_chain_design_example.py` 运行成功
- [ ] `python examples/joint_design_example.py` 运行成功
- [ ] `python examples/interface_recovery_example.py` 运行成功

## 📁 目录结构检查

### 预期目录结构
```
ComplexMPNN/
├── data/
│   ├── raw_pdb/          # (可选，不提交)
│   ├── processed/
│   │   ├── mpnn_pt/      # (可选，不提交)
│   │   ├── interface_masks/  # (可选，不提交)
│   │   └── structures/   # (可选，不提交)
│   └── splits/           # (可选，不提交)
├── checkpoints/          # (可选，不提交)
├── logs/                # (可选，不提交)
├── examples/            # ✅ 提交
├── *.py                 # ✅ 提交
├── *.sh                 # ✅ 提交
├── Makefile             # ✅ 提交
├── config.yaml          # ✅ 提交
├── environment.yml      # ✅ 提交
├── .gitignore           # ✅ 提交
├── README.md            # ✅ 提交
└── release_checklist.md # ✅ 提交
```

## 🔒 Git和发布检查

### Git仓库
- [ ] Git仓库已初始化 (`git init`)
- [ ] `.gitignore` 规则正确配置
- [ ] 大文件（data/、checkpoints/、logs/）未被track
- [ ] 所有代码文件已添加到Git (`git add .`)
- [ ] 首次提交已完成 (`git commit -m "Initial commit"`)

### 开源准备
- [ ] 选择开源许可证（MIT/Apache/GPL）
- [ ] 添加 LICENSE 文件
- [ ] 填写 README.md 中的联系方式
- [ ] 准备 GitHub Release 说明
- [ ] 准备项目截图/结果图

### 发布前最终检查
- [ ] 再次运行 `python check_full_pipeline.py`
- [ ] 再次运行 `bash run_all.sh`（在干净环境中）
- [ ] 检查所有代码中是否包含敏感信息
- [ ] 检查所有文件编码为UTF-8
- [ ] 检查所有脚本有执行权限（chmod +x）
- [ ] 检查README中所有链接有效

## 📝 文档完整性检查

### README.md
- [ ] 项目概述清晰
- [ ] 安装说明完整
- [ ] 使用说明详细
- [ ] 快速开始命令正确
- [ ] 项目结构说明清楚
- [ ] 结果说明完整
- [ ] 局限性说明诚实
- [ ] 贡献指南明确
- [ ] 许可证信息完整
- [ ] 致谢部分齐全

### 代码文档
- [ ] 所有Python文件有文件头注释
- [ ] 所有函数有docstring
- [ ] 关键代码有行内注释
- [ ] 示例代码有说明

## 🚀 发布后检查清单

### GitHub发布
- [ ] 创建 GitHub 仓库
- [ ] 推送代码到 GitHub
- [ ] 设置 GitHub Pages（如需）
- [ ] 添加项目标签（Topics）
- [ ] 设置仓库描述
- [ ] 创建 GitHub Release
- [ ] 上传附件（如需）

### 社区推广
- [ ] 分享到相关论坛/社区
- [ ] 撰写项目博客文章（如需）
- [ ] 准备演示视频（如需）
- [ ] 回复Issue和PR

## ✅ 完成状态

- [ ] 所有检查项已完成
- [ ] 项目可以正式开源发布

---

**最后更新**: 2026-03-09
**检查人**: AI开发工程师
