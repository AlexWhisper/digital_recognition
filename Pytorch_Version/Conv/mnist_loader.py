import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import os

def load_mnist_data(num_train: int = 5000, num_test: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用torchvision下载MNIST数据集
    
    Args:
        num_train: 训练样本数量
        num_test: 测试样本数量
        
    Returns:
        X_train: 训练图像数据，形状为 (num_train, 784)
        y_train: 训练标签，one-hot编码，形状为 (num_train, 10)
        X_test: 测试图像数据，形状为 (num_test, 784)
        y_test: 测试标签，one-hot编码，形状为 (num_test, 10)
    """
    try:
        import torchvision
        from torchvision import datasets, transforms
        
        # 创建数据目录
        data_dir = './mnist_data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 下载MNIST数据
        transform = transforms.Compose([transforms.ToTensor()])
        
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        
        # 提取数据并转换为PyTorch格式
        X_train = train_dataset.data.numpy()[:num_train].reshape(-1, 784).astype(np.float32) / 255.0
        y_train = train_dataset.targets.numpy()[:num_train]
        X_test = test_dataset.data.numpy()[:num_test].reshape(-1, 784).astype(np.float32) / 255.0
        y_test = test_dataset.targets.numpy()[:num_test]
        
        # 转换为one-hot编码
        y_train = to_one_hot(y_train)
        y_test = to_one_hot(y_test)
        
        print(f"MNIST数据加载成功: 训练{num_train}样本, 测试{num_test}样本")
        return X_train, y_train, X_test, y_test
        
    except ImportError:
        raise ImportError("请安装torchvision: pip install torchvision")
    except Exception as e:
        raise Exception(f"数据加载失败: {e}")

def to_one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """转换为one-hot编码"""
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def visualize_samples(X: np.ndarray, y: np.ndarray, num_samples: int = 16):
    """可视化样本"""
    true_labels = np.argmax(y, axis=1)
    
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, 16)):
        img = X[i].reshape(28,28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {true_labels[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试数据加载
    X_train, y_train, X_test, y_test = load_mnist_data(1000, 200)
    print(f"训练数据形状: {X_train.shape}")
    print(f"测试数据形状: {X_test.shape}")
    visualize_samples(X_train, y_train,num_samples=9)
