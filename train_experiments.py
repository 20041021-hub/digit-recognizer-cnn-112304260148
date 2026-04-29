import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class MNISTDataset(Dataset):
    def __init__(self, csv_file, is_test=False, transform=None):
        self.data = pd.read_csv(csv_file)
        self.is_test = is_test
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if not self.is_test:
            label = self.data.iloc[idx, 0]
            image = self.data.iloc[idx, 1:].values.astype(np.float32).reshape(28, 28)
            image = image / 255.0
            image = image[np.newaxis, ...]
            if self.transform:
                image = torch.tensor(image)
                image = self.transform(image)
                image = image.numpy()
            return torch.tensor(image), torch.tensor(label, dtype=torch.long)
        else:
            image = self.data.iloc[idx, :].values.astype(np.float32).reshape(28, 28)
            image = image / 255.0
            image = image[np.newaxis, ...]
            if self.transform:
                image = torch.tensor(image)
                image = self.transform(image)
                image = image.numpy()
            return torch.tensor(image)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def train_model(exp_name, optimizer_type='Adam', lr=0.001, batch_size=64, epochs=50, 
                use_augmentation=False, use_early_stopping=False, patience=5):
    
    if use_augmentation:
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
        ])
    else:
        transform = None
    
    full_train_dataset = MNISTDataset('train.csv', transform=transform)
    test_dataset = MNISTDataset('test.csv', is_test=True)
    
    dataset_size = len(full_train_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    best_val_loss = float('inf')
    early_stopping_count = 0
    
    print(f'=== 开始实验 {exp_name} ===')
    print(f'配置: optimizer={optimizer_type}, lr={lr}, batch_size={batch_size}, '
          f'augmentation={use_augmentation}, early_stopping={use_early_stopping}')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_epoch_loss = running_loss / len(train_loader)
        train_epoch_acc = 100 * correct / total
        
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
        
        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)
        train_acc_history.append(train_epoch_acc)
        val_acc_history.append(val_epoch_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_epoch_loss:.4f}, '
              f'Train Acc: {train_epoch_acc:.2f}%, Val Loss: {val_epoch_loss:.4f}, '
              f'Val Acc: {val_epoch_acc:.2f}%')
        
        if use_early_stopping:
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                early_stopping_count = 0
                torch.save(model.state_dict(), f'best_model_{exp_name}.pth')
            else:
                early_stopping_count += 1
                if early_stopping_count >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    model.load_state_dict(torch.load(f'best_model_{exp_name}.pth'))
                    break
    
    model.eval()
    test_correct = 0
    test_total = 0
    predictions = []
    
    with torch.no_grad():
        for images in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
    
    test_acc = 100 * test_correct / max(test_total, 1)
    
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv(f'submission_{exp_name}.csv', index=False, encoding='utf-8')
    
    print(f'=== 实验 {exp_name} 完成 ===')
    print(f'最终结果: Train Acc={train_acc_history[-1]:.2f}%, Val Acc={val_acc_history[-1]:.2f}%')
    print(f'最低Loss: {min(val_loss_history):.4f}, 收敛Epoch: {len(train_loss_history)}')
    print()
    
    return {
        'exp_name': exp_name,
        'train_acc': train_acc_history[-1],
        'val_acc': val_acc_history[-1],
        'test_acc': test_acc,
        'min_loss': min(val_loss_history),
        'converge_epoch': len(train_loss_history),
        'train_loss': train_loss_history,
        'val_loss': val_loss_history
    }

if __name__ == '__main__':
    experiments = [
        {'exp_name': 'Exp1', 'optimizer_type': 'SGD', 'lr': 0.01, 'batch_size': 64, 
         'use_augmentation': False, 'use_early_stopping': False},
        {'exp_name': 'Exp2', 'optimizer_type': 'Adam', 'lr': 0.001, 'batch_size': 64, 
         'use_augmentation': False, 'use_early_stopping': False},
        {'exp_name': 'Exp3', 'optimizer_type': 'Adam', 'lr': 0.001, 'batch_size': 128, 
         'use_augmentation': False, 'use_early_stopping': True},
        {'exp_name': 'Exp4', 'optimizer_type': 'Adam', 'lr': 0.001, 'batch_size': 64, 
         'use_augmentation': True, 'use_early_stopping': True}
    ]
    
    results = []
    for exp in experiments:
        result = train_model(**exp)
        results.append(result)
    
    print('=== 所有实验结果汇总 ===')
    print('| 实验编号 | Train Acc | Val Acc | Test Acc | 最低 Loss | 收敛 Epoch |')
    print('|----------|-----------|---------|----------|-----------|------------|')
    for r in results:
        print(f"| {r['exp_name']} | {r['train_acc']:.2f}% | {r['val_acc']:.2f}% | "
              f"{r['test_acc']:.2f}% | {r['min_loss']:.4f} | {r['converge_epoch']} |")
    
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'orange', 'green', 'red']
    for i, r in enumerate(results):
        plt.plot(range(1, len(r['train_loss'])+1), r['train_loss'], color=colors[i], 
                 linestyle='-', label=f'{r["exp_name"]} - Train')
        plt.plot(range(1, len(r['val_loss'])+1), r['val_loss'], color=colors[i], 
                 linestyle='--', label=f'{r["exp_name"]} - Val')
    
    plt.title('Training and Validation Loss for All Experiments')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_comparison.png')
    print('\nLoss对比图已保存为 loss_comparison.png')
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_results.csv', index=False, encoding='utf-8')
    print('实验结果已保存为 experiment_results.csv')