import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split  #分割数据集
from torch.optim.lr_scheduler import ReduceLROnPlateau # 引入调度器

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,padding=1)#保持特征图尺寸不变
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 128)#torch.lazy.Linear(128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64*7*7) #x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def main():
    # 数据集准备
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #其中download=True 是必须的，这样在第一次运行代码时，MNIST数据集会被自动下载到指定的目录（./data）。如果数据集已经存在，则不会重复下载。

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    #可以增加generator=torch.Generator().manual_seed(42)来保证没次分割的结果一致

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=4,drop_last=True)
    #num_workers 多线程处理数据，加快速度，drop_last=True 丢弃最后一批数据达不到Batch_size的部分，保持batch数据一样
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()#交叉熵作为损失函数，适用于多分类问题
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ================= 新增：初始化调度器和早停变量 =================
    # 调度器：连续 2 个 epoch 没提升，学习率减半

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)#调度器，缩小学习率

    num_epoches = 10
    best_val_acc = 0.0
    counter = 0          # 早停计数器
    patience = 5         # 早停耐心值 (必须大于调度器的 patience)

    # ================= 训练大循环 =================
    print("开始训练...")
    for epoch in range(num_epoches):
        # --- 1. 训练阶段 ---
        model.train()
        train_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()#训练阶段五个必备的环节，      零化梯度，前向传播，计算损失，反向传播，更新参数  
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epoches} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        # --- 2. 验证阶段 (注意缩进：必须在 for epoch 循环内部) ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():#验证和测试简短，没有计算梯度的环节
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"\n---> Epoch {epoch+1} Validation Accuracy: {val_acc:.2f}%")

        # --- 3. 调度器更新 (根据验证准确率调整学习率) ---
        current_lr = optimizer.param_groups[0]['lr']#获取学习率，但是没有使用optimizer.lr，因为optimizer.param可以获取不同神经网络层的学习率。
        scheduler.step(val_acc)#根据验证率修改学习率。
        if optimizer.param_groups[0]['lr'] < current_lr:
            print(f"📉 触发调度器：学习率已降至 {optimizer.param_groups[0]['lr']}")

        # --- 4. 早停与模型保存逻辑 --- 如果目前的验证率较高，则更新最佳验证率；如果没有提升，在五个计时器内仍然没有提升，那就停止训练。
        if val_acc > best_val_acc:
            print(f"✨ 发现更好的模型 ({best_val_acc:.2f}% -> {val_acc:.2f}%)，正在保存...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            counter = 0  # 表现有提升，重置计数器
        else:
            counter += 1
            print(f"⏳ 模型未提升，早停计数器: {counter} / {patience}")
            
            if counter >= patience:
                print("🛑 Early stopping triggered. 停止训练！")
                break  # 打断 for epoch 循环，提前结束
        print("-" * 50) # 打印分割线，让输出日志更好看
    # ================ 训练结束，开始测试集评估 =================
    print("\n开始测试集评估...")
    model.eval()
    correct=0
    totol=0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            totol += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100. * correct / totol
    print(f"测试集准确率: {test_acc:.2f}%")

    # 单张图片预测实例
    print("\n单张图片预测:")
    with torch.no_grad():
        outputs = model(images[0].unsqueeze(0).to(device))
        _,predicted = torch.max(outputs,1)
        print(f"预测类型:{predicted.item()},真实类型:{labels[0].item()}")
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()