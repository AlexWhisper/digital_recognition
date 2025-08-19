import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List

class RNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10, learning_rate: float = 0.001):
        """
        初始化循环神经网络
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: RNN层数
            num_classes: 输出类别数
            learning_rate: 学习率
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，形状为 (batch_size, 28, 28)
            
        Returns:
            输出预测
        """
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN前向传播
        out, _ = self.rnn(x, h0)
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int, 
                   batch_size: int = 32, verbose: bool = True) -> List[float]:
        """
        训练神经网络
        
        Args:
            X: 训练数据，形状为 (num_samples, 784)
            y: 标签，形状为 (num_samples,) 或 (num_samples, num_classes)
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练信息
            
        Returns:
            losses: 每轮的损失值
        """
        # 转换为PyTorch张量并重塑
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        
        # Reshape X to (num_samples, 28, 28) for RNN
        X_tensor = torch.FloatTensor(X).reshape(-1, 28, 28)
        y_tensor = torch.LongTensor(y)
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        self.train()  # 设置为训练模式
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                # 前向传播
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入数据，形状为 (num_samples, 784)
            
        Returns:
            预测结果，形状为 (num_samples, num_classes)
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).reshape(-1, 28, 28)
            outputs = self(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.numpy()
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 输入数据，形状为 (num_samples, 784)
            
        Returns:
            预测的类别索引
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).reshape(-1, 28, 28)
            outputs = self(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算准确率
        
        Args:
            X: 输入数据，形状为 (num_samples, 784)
            y: 真实标签（one-hot编码或类别索引）
            
        Returns:
            准确率
        """
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        
        predictions = self.predict_classes(X)
        return np.mean(predictions == y)


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = RNN()
    
    # 生成示例数据
    X = np.random.randn(1000, 784)
    y = np.random.randint(0, 10, 1000)
    
    # 训练模型
    losses = model.train_model(X, y, epochs=10, batch_size=32) # Reduced epochs for faster example
    
    # 预测
    predictions = model.predict_classes(X[:10])
    print(f"Predictions: {predictions}")
    
    # 计算准确率
    accuracy = model.accuracy(X, y)
    print(f"Accuracy: {accuracy:.4f}")
    
    # 保存模型 - 直接使用PyTorch方法
    torch.save(model.state_dict(), "rnn_model.pth")

