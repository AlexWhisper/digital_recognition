# 深度学习入门项目：手写数字识别

这是一个专为研0师弟师妹设计的深度学习入门项目，通过MNIST手写数字识别任务，循序渐进地学习从传统机器学习到现代深度学习的各种方法。项目包含多个版本的实现，从NumPy纯手工实现到PyTorch框架实现，再到模型部署，帮助初学者全面理解深度学习的原理和实践。

## 项目概述

本项目使用经典的MNIST手写数字识别数据集，实现了多种不同的神经网络架构：

- **Naive_Version**: 使用NumPy从零实现神经网络，理解底层原理
- **Pytorch_Version**: 使用PyTorch框架实现多种网络架构
  - MLP (多层感知机)
  - CNN (卷积神经网络)
  - RNN (循环神经网络)
  - Transformer (注意力机制)
- **Onnx_version**: 模型导出和部署，使用ONNX Runtime进行推理

## 学习路径建议

### 第一阶段：理解基础原理
1. 从 `Naive_Version` 开始，理解神经网络的前向传播和反向传播
2. 学习梯度下降优化算法的实现
3. 掌握损失函数和激活函数的作用

### 第二阶段：框架实践
1. 学习 `Pytorch_Version/MLP`，了解PyTorch基础用法
2. 进阶到 `Pytorch_Version/CNN`，理解卷积神经网络
3. 探索 `Pytorch_Version/RNN`，学习序列建模
4. 挑战 `Pytorch_Version/Transformer`，掌握注意力机制

### 第三阶段：模型部署
1. 学习 `Onnx_version`，了解模型导出和部署流程
2. 理解不同推理引擎的优缺点

## 项目结构

```
digital_recognition/
├── README.md                    # 项目总体说明
├── requirements.txt             # 统一依赖文件
├── .gitignore                   # Git忽略文件
│
├── Naive_Version/               # NumPy纯手工实现版本
│   ├── README.md               # 详细说明文档
│   ├── model.py                # 神经网络模型实现
│   ├── mnist_loader.py         # 数据加载器
│   ├── train_and_save.py       # 训练和保存脚本
│   ├── inference.py            # 推理脚本
│   └── requirements.txt        # 版本特定依赖
│
├── Pytorch_Version/             # PyTorch框架实现版本
│   ├── MLP/                    # 多层感知机实现
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── mnist_loader.py
│   │   ├── train_and_save.py
│   │   ├── inference.py
│   │   └── requirements.txt
│   │
│   ├── CNN/                    # 卷积神经网络实现
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── mnist_loader.py
│   │   ├── train_and_save.py
│   │   ├── inference.py
│   │   └── requirements.txt
│   │
│   ├── RNN/                    # 循环神经网络实现
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── mnist_loader.py
│   │   ├── train_and_save.py
│   │   ├── inference.py
│   │   └── requirements.txt
│   │
│   └── Transformer/            # Transformer实现
│       ├── README.md
│       ├── model.py
│       ├── mnist_loader.py
│       ├── train_and_save.py
│       ├── inference.py
│       └── requirements.txt
│
└── Onnx_version/               # ONNX模型导出和部署
    ├── export_onnx.py          # 模型导出脚本
    ├── inference.py            # ONNX推理脚本
    ├── mnist_loader.py         # 数据加载器
    ├── model.py                # 模型定义
    └── train_and_save.py       # 训练脚本
```

## 快速开始

### 环境准备

确保系统已安装Python 3.10，然后安装项目依赖：

```bash
# 安装基础依赖（适用于所有版本）
pip install -r requirements.txt

# 或者根据需要安装特定版本的依赖
cd Naive_Version && pip install -r requirements.txt
cd Pytorch_Version/CNN && pip install -r requirements.txt
```

### 运行示例

每个版本都包含完整的训练和推理流程：

```bash
# 以CNN版本为例
cd Pytorch_Version/CNN

# 1. 数据加载测试
python mnist_loader.py

# 2. 模型训练
python train_and_save.py

# 3. 模型推理
python inference.py
```

## 各版本特点对比

| 版本 | 实现方式 | 难度 | 学习重点 | 预期准确率 |
|------|----------|------|----------|------------|
| Naive_Version | NumPy手工实现 | ⭐⭐⭐ | 理解神经网络底层原理 | 85%+ |
| MLP | PyTorch框架 | ⭐⭐ | 学习PyTorch基础 | 90%+ |
| CNN | PyTorch框架 | ⭐⭐⭐ | 理解卷积操作 | 97%+ |
| RNN | PyTorch框架 | ⭐⭐⭐⭐ | 学习序列建模 | 97%+ |
| Transformer | PyTorch框架 | ⭐⭐⭐⭐⭐ | 掌握注意力机制 | 98%+ |
| ONNX | 模型部署 | ⭐⭐ | 学习模型部署 | 与原模型一致 |

## 技术栈

- **Python 3.7+**: 编程语言
- **NumPy**: 数值计算（Naive版本）
- **PyTorch**: 深度学习框架
- **Matplotlib**: 数据可视化
- **ONNX Runtime**: 模型推理引擎
- **Scikit-learn**: 机器学习工具

## 学习资源

- [MNIST数据集介绍](http://yann.lecun.com/exdb/mnist/)
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [深度学习基础理论](https://cs231n.github.io/)
- [神经网络与深度学习](https://nndl.github.io/)

## 常见问题

### Q: 应该从哪个版本开始学习？
A: 建议按照学习路径循序渐进：Naive_Version → MLP → CNN → RNN → Transformer → ONNX

### Q: 训练时间太长怎么办？
A: 可以减少训练轮数、减小批次大小或减少训练样本数量来加快训练速度

### Q: 如何提高模型准确率？
A: 可以尝试调整学习率、增加训练轮数、修改网络结构或使用数据增强技术

### Q: 遇到依赖安装问题怎么办？
A: 建议使用虚拟环境，并确保Python版本兼容。可以尝试使用conda或pip安装依赖

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！请确保：
1. 代码风格一致
2. 添加必要的注释
3. 更新相关文档
4. 测试代码功能

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

**注意**: 首次运行时会自动下载MNIST数据集（约11MB），请确保网络连接正常。
