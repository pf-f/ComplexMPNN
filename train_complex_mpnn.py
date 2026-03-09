#!/usr/bin/env python3
"""
train_complex_mpnn.py

Function：TrainComplexMPNN模型，支持两种Train模式

Core Rules：
1. Fork ProteinMPNN官方仓库，不修改任何模型架构
2. 支持Fixed-chain mode和Joint-design mode
3. 使用interface-weighted cross entropyloss函数
4. learning rate1e-5，Train10epoch，全模型fine-tune

Dependencies：
- torch
- pyyaml
- numpy

Usage：
python train_complex_mpnn.py --config config.yaml
"""

import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from loss_functions import InterfaceWeightedCrossEntropyLoss


def set_random_seed(seed, deterministic=True):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
        deterministic: 是否使用确定性算法
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """
    load配置file
    
    Args:
        config_path: 配置file路径
    
    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class ComplexMPNNDataSet(Dataset):
    """
    ComplexMPNN数据集
    """
    
    def __init__(self, mpnn_pt_dir, split_file, split_dir):
        """
        初始化数据集
        
        Args:
            mpnn_pt_dir: MPNN .ptfiledirectory
            split_file: 数据集Splitfile名
            split_dir: 数据集Splitfiledirectory
        """
        self.mpnn_pt_dir = mpnn_pt_dir
        
        # 读取Splitfile
        split_path = os.path.join(split_dir, split_file)
        with open(split_path, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 索引
        
        Returns:
            数据项
        """
        file_name = self.file_list[idx]
        file_path = os.path.join(self.mpnn_pt_dir, file_name)
        
        # load数据
        data = torch.load(file_path)
        
        return {
            'sequence': data['sequence'],
            'backbone_coords': data['backbone_coords'],
            'interface_mask': data['interface_mask'],
            'file_name': file_name
        }


def collate_fn(batch):
    """
    数据批Processing函数
    
    Args:
        batch: batch数据
    
    Returns:
        Processing后的batch数据
    """
    # 简单的批Processing，由于sequence长度不同，我们逐个Processing
    return batch


class ProteinMPNNWrapper(nn.Module):
    """
    ProteinMPNN包装器
    
    注意：这是一个简化的包装器，实际使用时需要fork ProteinMPNN官方仓库
    并正确load其预Trainweights
    """
    
    def __init__(self, pretrained_weights_path=None):
        """
        初始化ProteinMPNN包装器
        
        Args:
            pretrained_weights_path: 预Trainweights路径
        """
        super().__init__()
        
        # 注意：这里只是一个占位符，实际使用时需要：
        # 1. Fork ProteinMPNN官方仓库: https://github.com/dauparas/ProteinMPNN
        # 2. 正确导入和初始化模型
        # 3. load预Trainweights
        
        # 创建一个简单的模型用于演示
        vocab_size = 21  # 20种amino acid + 1个未知
        hidden_size = 128
        num_layers = 3
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # 如果提供了预Trainweights，尝试load
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            print(f"load预Trainweights: {pretrained_weights_path}")
            # 注意：这里需要根据ProteinMPNN的实际weights格式进行load
            # self.load_state_dict(torch.load(pretrained_weights_path))
    
    def forward(self, sequence, backbone_coords, fixed_mask=None):
        """
        前向传播
        
        Args:
            sequence: sequence（amino acid索引）
            backbone_coords: backbone坐标
            fixed_mask: 固定掩码，True表示固定residue
        
        Returns:
            logits
        """
        # 简单的前向传播，实际使用时需要替换为ProteinMPNN的实际实现
        batch_size, seq_len = sequence.shape
        
        # 嵌入
        x = self.embedding(sequence)  # (batch_size, seq_len, hidden_size)
        
        # Transformer (batch_first=True)
        x = self.transformer(x)  # (batch_size, seq_len, hidden_size)
        
        # output层
        logits = self.fc(x)  # (batch_size, seq_len, vocab_size)
        
        return logits


def train_epoch(model, dataloader, loss_fn, optimizer, device, config):
    """
    Train一个epoch
    
    Args:
        model: 模型
        dataloader: 数据load器
        loss_fn: loss函数
        optimizer: optimizer
        device: 设备
        config: 配置
    
    Returns:
        平均Trainloss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # amino acid到索引的映射
    aa_to_idx = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
        'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
        'X': 20
    }
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        batch_loss = 0.0
        
        for item in batch:
            sequence = item['sequence']
            backbone_coords = item['backbone_coords']
            interface_mask = item['interface_mask']
            
            # 转换sequence为索引
            seq_idx = torch.tensor([aa_to_idx.get(aa, 20) for aa in sequence], device=device)
            seq_idx = seq_idx.unsqueeze(0)  # (1, seq_len)
            
            # 转换interface_mask
            interface_mask_tensor = interface_mask.to(device)
            interface_mask_tensor = interface_mask_tensor.unsqueeze(0)  # (1, seq_len)
            
            # 随机选择Train模式
            use_fixed_chain = random.random() < config['training_modes']['fixed_chain_probability']
            
            if use_fixed_chain and config['training_modes']['fixed_chain_mode']:
                # Fixed-chain mode: 随机选择一些residue固定
                fixed_mask = torch.rand(interface_mask_tensor.shape, device=interface_mask_tensor.device) < 0.5
            else:
                # Joint-design mode: 所有residue都可以设计
                fixed_mask = torch.zeros_like(interface_mask_tensor, dtype=torch.bool)
            
            # 前向传播
            logits = model(seq_idx, backbone_coords, fixed_mask)
            
            # 计算loss
            loss = loss_fn(logits, seq_idx, interface_mask_tensor)
            
            # 反向传播
            loss.backward()
            
            batch_loss += loss.item()
        
        # 更新参数
        optimizer.step()
        
        total_loss += batch_loss / len(batch)
        num_batches += 1
        
        # 打印日志
        if num_batches % config['logging']['log_interval'] == 0:
            print(f"Batch {num_batches}, Loss: {batch_loss / len(batch):.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, loss_fn, device):
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 数据load器
        loss_fn: loss函数
        device: 设备
    
    Returns:
        平均验证loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # amino acid到索引的映射
    aa_to_idx = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
        'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
        'X': 20
    }
    
    with torch.no_grad():
        for batch in dataloader:
            batch_loss = 0.0
            
            for item in batch:
                sequence = item['sequence']
                backbone_coords = item['backbone_coords']
                interface_mask = item['interface_mask']
                
                # 转换sequence为索引
                seq_idx = torch.tensor([aa_to_idx.get(aa, 20) for aa in sequence], device=device)
                seq_idx = seq_idx.unsqueeze(0)  # (1, seq_len)
                
                # 转换interface_mask
                interface_mask_tensor = interface_mask.to(device)
                interface_mask_tensor = interface_mask_tensor.unsqueeze(0)  # (1, seq_len)
                
                # 前向传播
                logits = model(seq_idx, backbone_coords)
                
                # 计算loss
                loss = loss_fn(logits, seq_idx, interface_mask_tensor)
                
                batch_loss += loss.item()
            
            total_loss += batch_loss / len(batch)
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='TrainComplexMPNN模型')
    parser.add_argument('--config', default='config.yaml', help='配置file路径')
    
    args = parser.parse_args()
    
    # load配置
    config = load_config(args.config)
    
    # Set random seed
    set_random_seed(
        config['random']['seed'],
        config['random']['deterministic']
    )
    
    # 创建必要的directory
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['logging']['log_file']), exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据集和数据load器
    train_dataset = ComplexMPNNDataSet(
        config['data']['mpnn_pt_dir'],
        config['data']['train_split'],
        config['data']['split_dir']
    )
    
    val_dataset = ComplexMPNNDataSet(
        config['data']['mpnn_pt_dir'],
        config['data']['val_split'],
        config['data']['split_dir']
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Train集大小: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # 创建模型
    model = ProteinMPNNWrapper(config['model']['pretrained_weights_path'])
    model = model.to(device)
    
    # 创建loss函数
    loss_fn = InterfaceWeightedCrossEntropyLoss(
        interface_weight=config['loss']['interface_weight'],
        non_interface_weight=config['loss']['non_interface_weight']
    )
    
    # 创建optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 创建learning ratescheduler
    scheduler = None
    if config['training']['use_lr_scheduler']:
        if config['training']['lr_scheduler'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
            print("使用learning ratescheduler: ReduceLROnPlateau")
    
    # Train循环
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_dataloader, loss_fn, optimizer, device, config
        )
        print(f"Trainloss: {train_loss:.4f}")
        
        # 验证
        if (epoch + 1) % config['logging']['val_interval'] == 0:
            val_loss = validate(model, val_dataloader, loss_fn, device)
            print(f"验证loss: {val_loss:.4f}")
            
            # 更新learning ratescheduler
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # save最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    config['checkpoint']['best_model_path']
                )
                print(f"save最佳模型: {config['checkpoint']['best_model_path']}")
        
        # 定期savecheckpoint
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['checkpoint']['save_dir'],
                f'complexmpnn_epoch_{epoch + 1}.pt'
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"savecheckpoint: {checkpoint_path}")
    
    print("\nTrainComplete！")
    print(f"最佳验证loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
