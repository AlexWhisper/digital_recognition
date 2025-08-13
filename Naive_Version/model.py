import numpy as np
from typing import List

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
        """
        初始化神经网络（仅用numpy实现）
        Args:
            layer_sizes: 每层神经元数量，例如 [784, 128, 64, 10]
            learning_rate: 学习率
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        # 权重和偏置初始化
        self.weights = [np.random.randn(n_in, n_out) * np.sqrt(2. / n_in)
                        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((1, n_out)) for n_out in layer_sizes[1:]]

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, x):
        activations = [x]
        pre_activations = []
        for i in range(self.num_layers - 2):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = self.relu(z)
            activations.append(a)
        # 最后一层
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        a = self.softmax(z)
        activations.append(a)
        return activations, pre_activations

    def compute_loss(self, y_pred, y_true):
        # y_true: (batch, num_classes) 或 (batch,)
        if len(y_true.shape) == 1:
            y_true = np.eye(self.layer_sizes[-1])[y_true]
        # 防止log(0)
        eps = 1e-8
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

    def backward(self, activations, pre_activations, y_true):
        grads_w = [None] * (self.num_layers - 1)
        grads_b = [None] * (self.num_layers - 1)
        batch_size = y_true.shape[0]
        # y_true: (batch, num_classes) 或 (batch,)
        if len(y_true.shape) == 1:
            y_true = np.eye(self.layer_sizes[-1])[y_true]
        # 输出层梯度
        delta = (activations[-1] - y_true) / batch_size
        grads_w[-1] = activations[-2].T @ delta
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)
        # 反向传播隐藏层
        for l in range(self.num_layers - 2, 0, -1):
            delta = (delta @ self.weights[l].T) * self.relu_deriv(pre_activations[l-1])
            grads_w[l-1] = activations[l-1].T @ delta
            grads_b[l-1] = np.sum(delta, axis=0, keepdims=True)
        return grads_w, grads_b

    def update_params(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 32, verbose: bool = True):
        losses = []
        n = X.shape[0]
        for epoch in range(epochs):
            # 打乱数据
            idx = np.random.permutation(n)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            epoch_loss = 0
            num_batches = 0
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                activations, pre_activations = self.forward(X_batch)
                loss = self.compute_loss(activations[-1], y_batch)
                grads_w, grads_b = self.backward(activations, pre_activations, y_batch)
                self.update_params(grads_w, grads_b)
                epoch_loss += loss
                num_batches += 1
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return activations[-1]

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict_classes(X)
        if len(y.shape) == 2:
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y
        return np.mean(y_pred == y_true)

    def save(self, path_prefix):
        np.savez(path_prefix + '_weights.npz', *self.weights)
        np.savez(path_prefix + '_biases.npz', *self.biases)
        with open(path_prefix + '_structure.json', 'w') as f:
            import json
            json.dump({'layer_sizes': self.layer_sizes, 'learning_rate': self.learning_rate}, f)

    @staticmethod
    def load(path_prefix):
        with open(path_prefix + '_structure.json', 'r') as f:
            import json
            info = json.load(f)
        net = NeuralNetwork(info['layer_sizes'], info['learning_rate'])
        weights = np.load(path_prefix + '_weights.npz')
        biases = np.load(path_prefix + '_biases.npz')
        net.weights = [weights[f'arr_{i}'] for i in range(len(net.layer_sizes)-1)]
        net.biases = [biases[f'arr_{i}'] for i in range(len(net.layer_sizes)-1)]
        return net


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
    
    # 保存模型 
    model.save("neural_network")

