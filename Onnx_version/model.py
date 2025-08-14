import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List

class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
        """
        初始化神经网络
        
        Args:
            layer_sizes: 每层神经元数量，例如 [784, 128, 64, 10] 表示输入784维，两个隐藏层，输出10维
            learning_rate: 学习率
        """
        super(NeuralNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # 构建网络层
        self.layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，形状为 (batch_size, input_size)
            
        Returns:
            输出预测
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        
        # 最后一层不使用激活函数，因为CrossEntropyLoss内部会处理
        x = self.layers[-1](x)
        return x
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int, 
                   batch_size: int = 32, verbose: bool = True) -> List[float]:
        """
        训练神经网络
        
        Args:
            X: 训练数据，形状为 (num_samples, input_size)
            y: 标签，形状为 (num_samples,) 或 (num_samples, num_classes)
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练信息
            
        Returns:
            losses: 每轮的损失值
        """
        # 转换为PyTorch张量
        if len(y.shape) == 2:
            # 如果y是one-hot编码，转换为类别索引
            y = np.argmax(y, axis=1)
        X_tensor = torch.FloatTensor(X)
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
            X: 输入数据，形状为 (num_samples, input_size)
            
        Returns:
            预测结果，形状为 (num_samples, num_classes)
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.numpy()
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 输入数据，形状为 (num_samples, input_size)
            
        Returns:
            预测的类别索引
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算准确率
        
        Args:
            X: 输入数据，形状为 (num_samples, input_size)
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
    model = NeuralNetwork([784, 128, 64, 10])
    
    # 生成示例数据
    X = np.random.randn(1000, 784)
    y = np.random.randint(0, 10, 1000)
    
    # 训练模型
    losses = model.train_model(X, y, epochs=50, batch_size=32)
    
    # 预测
    predictions = model.predict_classes(X[:10])
    print(f"Predictions: {predictions}")
    
    # 计算准确率
    accuracy = model.accuracy(X, y)
    print(f"Accuracy: {accuracy:.4f}")
    
    # 保存模型 - 直接使用PyTorch方法
    torch.save(model.state_dict(), "neural_network.pth")

