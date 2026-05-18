# dataset.py
import torch
import logging
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def get_mnist_loaders(config):
    """
    根据配置字典，自动下载、划分并返回 train, val, test 的 DataLoader
    """
    # 1. 定义数据预处理流
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 2. 加入try,except块捕获数据加载异常
    try:
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        logging.error(f"❌ 数据集下载或加载失败: {e}")
        raise e  # 将异常向上抛出，让主程序知道
        
    # 3. 划分验证集（使用配置文件中的固定种子）
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    # 4. 通过 DataLoader 打包成多线程货车编队
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    logging.info(f"📊 数据就绪 -> 训练批次: {len(train_loader)} | 验证批次: {len(val_loader)} | 测试批次: {len(test_loader)}")
    return train_loader, val_loader, test_loader