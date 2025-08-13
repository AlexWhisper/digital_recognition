# MNIST 手写数字识别神经网络项目（NumPy 版）

这是一个使用 NumPy 从零实现的 MNIST 手写数字识别项目，包含从数据加载、模型定义、训练到推理的完整流程，适合教学中展示神经网络的底层原理（前向传播、反向传播与参数更新）。

## 🎯 项目概述

本项目实现了一个多层感知机（MLP）神经网络来识别手写数字（0-9），使用经典的 MNIST 数据集进行训练和测试。实现仅依赖标准库、NumPy 与 Matplotlib，无需安装 PyTorch 或其他深度学习框架。

## 📁 项目结构

```
Naive_Version/
├── model.py                # 神经网络（NumPy 实现）
├── mnist_loader.py         # MNIST 数据下载与加载（纯标准库 + NumPy）
├── train_and_save.py       # 模型训练、评估与保存
├── inference.py            # 模型加载与推理展示
├── requirements.txt        # 最小依赖（numpy、matplotlib）
├── README.md               # 项目说明文档
└── saved_models/           # 训练后保存的模型（运行后生成）
    ├── model_structure.json
    ├── model_weights.npz
    └── model_biases.npz
```

## 🚀 快速开始

### 1. 环境准备

确保系统已安装 Python 3.7+ 和 pip，然后安装项目依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行顺序

#### 步骤 1: 数据加载测试
```bash
python mnist_loader.py
```
- 自动下载并解析 MNIST 数据集（约 11MB）
- 测试数据加载功能
- 可视化样本数据

#### 步骤 2: 模型训练
```bash
python train_and_save.py
```
- 加载 MNIST 数据
- 创建并训练神经网络（NumPy 版）
- 评估模型性能
- 保存训练好的模型参数与结构

#### 步骤 3: 模型推理
```bash
python inference.py
```
- 加载保存的模型
- 进行预测推理
- 可视化预测结果

## 🧠 模型架构

### 神经网络结构
- 输入层: 784 个神经元（28×28 像素图像展平）
- 隐藏层 1: 128 个神经元，ReLU 激活函数
- 隐藏层 2: 64 个神经元，ReLU 激活函数
- 输出层: 10 个神经元（对应数字 0-9），Softmax 概率输出

### 训练参数（默认）
- 优化器: SGD（随机梯度下降）
- 学习率: 0.001
- 损失函数: 交叉熵（one-hot 标签）
- 批次大小: 32
- 训练轮数: 30

## 📊 功能特性

### 数据加载 (`mnist_loader.py`)
- 纯标准库下载与解析 MNIST（二进制 IDX 格式）
- 数据标准化至 [0,1]
- 标签 one-hot 编码
- 样本可视化

### 模型定义 (`model.py`)
- 纯 NumPy 实现前向传播与反向传播
- ReLU 与 Softmax 支持
- 训练、预测与准确率计算
- 模型保存与加载（结构 JSON + 权重/偏置 NPZ）

### 训练流程 (`train_and_save.py`)
- 完整的训练流程
- 训练过程可视化（损失曲线）
- 性能评估与分析（含简单混淆矩阵片段）
- 模型自动保存至 `saved_models/`

### 推理应用 (`inference.py`)
- 模型加载与恢复
- 单张图像预测
- 预测结果可视化与交互

## 📈 预期表现

在标准 MNIST 测试集上，简单的两层 MLP 通常可达到：
- 训练准确率: 90%+（与超参、样本量有关）
- 测试准确率: 85%+（NumPy 实现，非框架加速）

## 🛠️ 自定义配置

### 修改网络结构
在 `train_and_save.py` 中修改 `layer_sizes`：
```python
layer_sizes = [784, 256, 128, 64, 32, 10]
```

### 调整训练参数
```python
learning_rate = 0.001
epochs = 50
batch_size = 64
```

### 更改数据集大小
```python
X_train, y_train, X_test, y_test = load_mnist_data(
    num_train=10000,
    num_test=2000
)
```

## 🐛 常见问题

### Q: 首次运行会下载数据吗？
A: 会。`mnist_loader.py` 会自动从公开镜像下载 MNIST（约 11MB），并缓存在 `./mnist_data/`。

### Q: 还需要安装 PyTorch / Torchvision 吗？
A: 不需要。本项目为纯 NumPy 实现，仅需 `numpy` 与 `matplotlib`。

### Q: 训练较慢怎么办？
A: 可减少训练轮数、减小批次大小或减少训练样本数量以加快体验式教学。

## 📚 学习资源

- [MNIST 数据集介绍](http://yann.lecun.com/exdb/mnist/)
- [神经网络基础理论](https://cs231n.github.io/)

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

> 注意: 首次运行时会自动下载 MNIST 数据集，请确保网络连接可用。
