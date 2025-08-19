import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size=28, d_model=128, nhead=4, num_encoder_layers=2, num_classes=10, learning_rate: float = 0.001):
        """
        初始化Transformer模型
        
        Args:
            input_size: 输入特征维度
            d_model: 模型维度
            nhead: 多头注意力机制的头数
            num_encoder_layers: Transformer编码器层数
            num_classes: 输出类别数
            learning_rate: 学习率
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.encoder = nn.Linear(input_size, d_model)
        self.fc = nn.Linear(d_model, num_classes)
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        '''
        我们直到经典的transformer{尤其是在机器翻译等任务中}包含一个编码器{Encoder}和一个解码器{Decoder}。
        编码器{Encoder} 的作用是理解和学习输入序列的深层表示。
        解码器{Decoder} 的作用是基于编码器学到的表示来生成一个输出序列。
        然而，在这个项目中，我们处理的是 图像分类 任务{MNIST手写数字识别}，而不是序列到序列的转换任务。
        对于分类任务，我们的目标不是生成一个新的序列，而是将输入{在这里是图像}归类到一个预定义的类别中{0到9}。
        因此，我们只需要 编码器 部分就足够了。
        '''
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            X: 输入数据，形状为 (batch_size, seq_len, input_size)
            
        Returns:
            输出预测
        """
        X = self.encoder(X) * math.sqrt(self.d_model)
        X = self.pos_encoder(X.permute(1, 0, 2))
        output = self.transformer_encoder(X)
        output = self.fc(output.mean(dim=0))
        return output

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
        
        # Reshape X to (num_samples, 28, 28) for Transformer
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
            y: 真实标签{one-hot编码或类别索引}
            
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
    model = TransformerModel()
    
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
    torch.save(model.state_dict(), "transformer_model.pth")

