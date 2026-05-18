# train.py
import logging
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix
import os

# 同时导入模型与数据加载器
from model import SimpleCNN
from dataset import get_mnist_loaders

# 全局配置字典
CONFIG = {
    'batch_size': 64,
    'lr': 0.001,
    'epochs': 10,
    'early_stop_patience': 5,
    'scheduler_patience': 2,
    'factor': 0.5,
    'seed': 42
}

# 配置标准规范日志
logging.basicConfig(
    level=logging.INFO,#高于info才能被记录
    format='%(asctime)s - %(levelname)s - %(message)s',
    #asctime获取时间，levelment自动输入logging的等级，
    handlers=[
        logging.StreamHandler(),#数据传去终端
        logging.FileHandler('training.log', encoding='utf-8')#数据传向本地磁盘，及train.log
    ]
)

def main():
    # 设置随机种子和计算设备
    torch.manual_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"🖥️ 当前运行设备: {device}")

    # 1. 呼叫数据流水线获取数据
    try:
        train_loader, val_loader, test_loader = get_mnist_loaders(CONFIG)
    except Exception:
        logging.error("程序因数据准备失败而强行终止。")
        return

    # 2. 初始化网络与核心组件
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=CONFIG['factor'], patience=CONFIG['scheduler_patience'])

    best_val_acc = 0.0
    early_stop_counter = 0          
    
    # 用于可视化的收敛曲线记录器
    history = {'train_loss': [], 'val_acc': []}

    logging.info("🚀 架构组装完毕，正式启动模型训练大循环...")
    
    for epoch in range(CONFIG['epochs']):
        # --- 训练阶段 ---
        model.train()
        epoch_train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad() # 1.零化梯度
            outputs = model(images) # 2.前向传播
            loss = criterion(outputs, labels) # 3.计算损失
            loss.backward() # 4.反向传播
            optimizer.step() # 5.更新参数
            
            epoch_train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                logging.info(f"Epoch {epoch+1}/{CONFIG['epochs']} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        history['train_loss'].append(epoch_train_loss / len(train_loader))

        # --- 验证阶段 ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100. * val_correct / val_total
        history['val_acc'].append(val_acc)
        logging.info(f"--> Epoch {epoch+1} 验证集准确率: {val_acc:.2f}%")

        # --- 学习率调度 ---
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        if optimizer.param_groups[0]['lr'] < current_lr:
            logging.warning(f"📉 触发学习率衰减：学习率已降至 {optimizer.param_groups[0]['lr']}")

        # --- 早停与最优模型保存 ---
        if val_acc > best_val_acc:
            logging.info(f"✨ 发现更好的模型 ({best_val_acc:.2f}% -> {val_acc:.2f}%)，权重已更新保存。")
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            early_stop_counter = 0  
        else:
            early_stop_counter += 1
            logging.info(f"⏳ 模型未提升，早停计数器: {early_stop_counter} / {CONFIG['early_stop_patience']}")
            
            if early_stop_counter >= CONFIG['early_stop_patience']:
                logging.warning("🛑 触发早停条件，提前终止训练流程！")
                break
        logging.info("-" * 50)

    # ================ 训练结束：可视化收敛图像 =================
    logging.info("📊 正在绘制并收敛曲线...")
    plt.figure(figsize=(12, 5))#给图像画布 12*5
    plt.subplot(1, 2, 1)#subplot 分配位置 1行2列第1个位置
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], 'bo-', label='Train Loss')
    #x轴epoch,y轴train_loss
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Convergence (Loss)')
    plt.grid(True)#grid网格
    plt.legend()#图注

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], 'ro-', label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.grid(True)
    plt.legend()#图注
    plt.tight_layout()#自动排版
    plt.show()
    # ================ 最终大考：sklearn 进阶指标评估 =================
    logging.info("🎓 载入历史最佳权重，开始进行全量测试集终极评估 (sklearn)...")
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    except Exception as e:
        logging.error(f"❌ 最佳权重文件加载失败: {e}")
        return
        
    model.eval()
    all_preds, all_labels = [], []#模型预测和标准标签放进列表
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)#依旧提取最大分数的那个数字
            all_preds.extend(predicted.cpu().numpy())
            #numpy()将tensor转为numpy,cpu()将数据从GPU中转回，extend()将数组中的64个数字单独提取出来
            all_labels.extend(labels.cpu().numpy())#同上

   #利用classification_report，recall,F1-score以及支持率
    logging.info("\n" + "="*20 + " sklearn 详细分类报告 " + "="*20 + "\n" + 
                 classification_report(all_labels, all_preds, digits=4))
    #输出precision、recall、F1-score、support,保留四位小数，其中precision是预测对的占预测总数之比，
    # recall是预测对的占实际综述之比，F1是前两个的调和平均，support是每个类别数量
    #利用confusion_matrix,输出混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)#对角线是预测正确的，非对角线是吧i预测为j的
    logging.info(f"\n混淆矩阵对照表:\n{cm}")

    # ================ 单张图片盲测 =================
    logging.info("🎯 抽样单张真实图片进行推理验证:")
    with torch.no_grad():
        test_iter = iter(test_loader)
        test_images, test_labels = next(test_iter)
        single_image = test_images[0].unsqueeze(0).to(device)
        single_label = test_labels[0]
        
        outputs = model(single_image)
        _, predicted = torch.max(outputs, 1)
        logging.info(f"【抽查结果】-> 模型预测类型: {predicted.item()} | 真实类型: {single_label.item()}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()