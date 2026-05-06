import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# 定义数据集类
class MNISTDataset(Dataset):
    def __init__(self, csv_file, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.is_test = is_test
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if not self.is_test:
            # 训练数据包含label
            label = self.data.iloc[idx, 0]
            image = self.data.iloc[idx, 1:].values.astype(np.float32) / 255.0  # 归一化
            return torch.tensor(image), torch.tensor(label, dtype=torch.long)
        else:
            # 测试数据没有label
            image = self.data.iloc[idx, :].values.astype(np.float32) / 255.0  # 归一化
            return torch.tensor(image)

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积：输入通道1，输出通道16，卷积核3x3
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # 激活函数
        self.relu1 = nn.ReLU()
        # 池化层：2x2
        self.pool1 = nn.MaxPool2d(2)
        # 第二层卷积：输入通道16，输出通道32，卷积核3x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 激活函数
        self.relu2 = nn.ReLU()
        # 池化层：2x2
        self.pool2 = nn.MaxPool2d(2)
        # 全连接层1：输入特征数32*7*7，输出128
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # 激活函数
        self.relu3 = nn.ReLU()
        # 全连接层2：输入128，输出10
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # 调整输入形状：(batch_size, 784) → (batch_size, 1, 28, 28)
        x = x.view(-1, 1, 28, 28)
        # 第一层卷积
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # 第二层卷积
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # 展平特征图
        x = x.view(-1, 32 * 7 * 7)
        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 训练和测试函数
def train_and_test():
    # 超参数
    batch_size = 128
    learning_rate = 0.001
    epochs = 50
    validation_split = 0.2  # 验证集比例
    
    # 加载数据
    full_train_dataset = MNISTDataset('train.csv')
    test_dataset = MNISTDataset('test.csv', is_test=True)
    
    # 划分训练集和验证集
    dataset_size = len(full_train_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 记录loss历史
    train_loss_history = []
    val_loss_history = []
    
    # 训练模型
    print('开始训练...')
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_epoch_loss = running_loss / len(train_loader)
        train_epoch_acc = 100 * correct / total
        train_loss_history.append(train_epoch_loss)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total
        val_loss_history.append(val_epoch_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_acc:.2f}%, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}%')
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist_model.pth')
    print('\n模型已保存为 mnist_model.pth')
    
    # 绘制loss变化图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_loss_history, 'b-', label='Training Loss')
    plt.plot(range(1, epochs+1), val_loss_history, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    print('\nLoss变化图已保存为 loss_plot.png')
    
    # 测试模型并生成预测
    print('\n开始测试并生成预测...')
    model.eval()  # 设置为评估模式
    predictions = []
    
    with torch.no_grad():
        for images in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
    
    # 生成sample_submission.csv
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    # 确保编码正确，不包含BOM
    submission.to_csv('sample_submission.csv', index=False, encoding='utf-8')
    print('\n预测完成，已生成sample_submission.csv文件')

if __name__ == '__main__':
    train_and_test()