# MNIST 手写数字识别Transformer项目

这是一个使用 PyTorch 实现的完整 MNIST 手写数字识别项目，包含从数据加载、模型定义、训练到推理的完整流程。

## 项目概述

本项目实现了一个Transformer模型来识别手写数字（0-9），使用经典的 MNIST 数据集进行训练和测试。项目采用模块化设计，代码结构清晰，适合学习和理解Transformer架构和注意力机制的基本原理。

## 项目结构

```
Transformer/
├── model.py                # Transformer模型定义
├── mnist_loader.py         # MNIST 数据加载和预处理
├── train_and_save.py       # 模型训练和保存
├── inference.py            # 模型加载和推理
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明文档
├── saved_models/           # 保存的模型文件
│   ├── transformer_model.pth # PyTorch 模型权重
│   └── model_structure.json # 模型结构信息
└── mnist_data/             # MNIST 数据集（自动下载）
```

## 快速开始

### 1. 环境准备

确保你的系统已安装 Python 3.7+ 和 pip，然后安装项目依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行顺序

项目文件需要按以下顺序运行：

#### 步骤 1: 数据加载测试
```bash
python mnist_loader.py
```
- 下载 MNIST 数据集
- 测试数据加载功能
- 可视化样本数据

#### 步骤 2: 模型训练
```bash
python train_and_save.py
```
- 加载 MNIST 数据
- 创建并训练Transformer模型
- 评估模型性能
- 保存训练好的模型

#### 步骤 3: 模型推理
```bash
python inference.py
```
- 加载保存的模型
- 进行预测推理
- 可视化预测结果

## 模型架构

### Transformer 结构
- **输入**: 28x28 的灰度图像，视为一个包含28个时间步的序列，每个时间步有28个特征
- **位置编码**: 为序列添加位置信息
- **Transformer编码器**: 2层编码器，4个注意力头，模型维度128
- **多头自注意力机制**: 学习序列内部的依赖关系
- **前馈神经网络**: 每个编码器层包含前馈网络
- **全连接层**: 128个神经元 -> 10个神经元 (对应0-9)
- **输出层**: 10 个神经元（对应数字 0-9）

### 训练参数
- **优化器**: Adam
- **学习率**: 0.001
- **损失函数**: CrossEntropyLoss
- **批次大小**: 32
- **训练轮数**: 50

## 功能特性

### 数据加载 (`mnist_loader.py`)
- 自动下载 MNIST 数据集
- 数据预处理和标准化
- 样本可视化
- One-hot 编码转换

### 模型定义 (`model.py`)
- Transformer架构实现
- 位置编码模块
- 多头自注意力机制
- 内置训练方法
- 预测和评估功能
- 模型保存和加载

### 训练流程 (`train_and_save.py`)
- 完整的训练流程
- 训练过程可视化
- 性能评估和分析
- 模型自动保存

### 推理应用 (`inference.py`)
- 模型加载和恢复
- 单张图像预测
- 预测结果可视化
- 交互式测试模式

## 技术实现

### 核心技术
- **PyTorch**: 深度学习框架
- **NumPy**: 数值计算
- **Matplotlib**: 数据可视化
- **Torchvision**: 数据集加载

### 关键特性
- 模块化设计，代码复用性高
- 完整的错误处理和异常管理
- 详细的训练过程监控
- 灵活的模型配置选项

## 性能表现

在标准 MNIST 测试集上，模型通常能达到：
- **训练准确率**: 98%+
- **测试准确率**: 97%+
- **训练时间**: 约 2-5 分钟（取决于硬件配置）

## 自定义配置

### 调整训练参数
在 `train_and_save.py` 中修改训练相关的超参数：

```python
learning_rate = 0.001      # 学习率
epochs = 50                # 训练轮数
batch_size = 64            # 批次大小
```

### 更改数据集大小
调整训练和测试样本数量：

```python
X_train, y_train, X_test, y_test = load_mnist_data(
    num_train=10000,    # 训练样本数
    num_test=2000       # 测试样本数
)
```

## 常见问题

### Q: 运行时提示 "No module named 'torch'"
**A**: 请确保已正确安装 PyTorch：
```bash
pip install torch torchvision
```

### Q: 训练过程中内存不足
**A**: 可以减少批次大小或训练样本数量：
```python
batch_size = 16  # 减小批次大小
num_train = 2000 # 减少训练样本
```

### Q: 模型保存失败
**A**: 确保 `saved_models/` 目录存在写入权限，或手动创建该目录。

## 学习资源

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [MNIST 数据集介绍](http://yann.lecun.com/exdb/mnist/)
- [Transformer架构详解](https://arxiv.org/abs/1706.03762)
- [注意力机制原理](https://distill.pub/2016/augmented-rnns/)

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

**注意**: 首次运行时会自动下载 MNIST 数据集（约 11MB），请确保网络连接正常。
