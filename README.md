# hello_fxb_world
feature1-1
## 全网同名：方向标studyhard
### 小红书：方向标studyhard
### bilibili：方向标studyhard
### 视频号：方向标studyhard



## 示例代码：
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 1. Hyperparameters
input_size = 28       # 每行 28 像素
sequence_length = 28  # 一张图片 28 行
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 1
learning_rate = 0.001

# 2. MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 3. LSTM 模型定义
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, 1, 28, 28) → reshape 为 (batch, 28, 28)
        x = x.reshape(-1, sequence_length, input_size)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # 取序列最后一个时间步输出
        out = self.fc(out[:, -1, :])
        return out

model = LSTMNet(input_size, hidden_size, num_layers, num_classes)

# 4. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练过程
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. 测试
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
```
