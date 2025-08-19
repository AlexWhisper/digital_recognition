#!/usr/bin/env python3
"""
MNIST手写体识别完整示例
使用PyTorch实现的神经网络进行手写数字识别
"""
import numpy as np
import matplotlib.pyplot as plt
from model import TransformerModel
from mnist_loader import load_mnist_data, visualize_samples
import time
import json
import os
import torch

def save_model_info(network, model_dir='./saved_models'):
    """
    Save neural network model structure and PyTorch state
    """
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save network structure for Transformer
    model_info = {
        'input_size': network.encoder.in_features,
        'd_model': network.d_model,
        'nhead': network.transformer_encoder.layers[0].self_attn.num_heads,
        'num_encoder_layers': network.transformer_encoder.num_layers,
        'num_classes': network.fc.out_features,
        'learning_rate': network.optimizer.param_groups[0]['lr']
    }
    
    with open(os.path.join(model_dir, 'model_structure.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save PyTorch model state
    model_path = os.path.join(model_dir, 'transformer_model.pth')
    torch.save(network.state_dict(), model_path)
    
    print(f"Model saved to {model_dir}/")
    print(f"   - Model structure: model_structure.json")
    print(f"   - PyTorch model: transformer_model.pth")

def main():
    print("=" * 60)
    print("MNIST Handwritten Digit Recognition Transformer Demo (PyTorch)")
    print("=" * 60)
    
    # 1. Load MNIST data
    print("\n1. Loading MNIST dataset...")

    # For demonstration, use smaller dataset
    X_train, y_train, X_test, y_test = load_mnist_data(
        num_train=5000,  # Training samples
        num_test=1000    # Test samples
    )
    
    # Data is already in PyTorch-compatible format: (num_samples, features)
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # # 2. Visualize some training samples
    # print("\n2. Visualizing training samples...")
    # visualize_samples(X_train, y_train, num_samples=9)
    
    # 3. Create neural network
    print("\n3. Creating neural network...")
    learning_rate = 0.001
    
    network = TransformerModel(input_size=28, d_model=128, nhead=4, num_encoder_layers=2, num_classes=10, learning_rate=learning_rate)
    print(f"Network structure: input_size={network.encoder.in_features}, d_model={network.d_model}, nhead={network.transformer_encoder.layers[0].self_attn.num_heads}, num_encoder_layers={network.transformer_encoder.num_layers}, num_classes={network.fc.out_features}")
    
    # 4. Train network
    print("\n4. Starting training...")
    print(f"  - Training samples: {X_train.shape[0]}")
    
    losses = network.train_model(
        X_train, y_train, 
        epochs=50, 
        batch_size=32, 
        verbose=True
    )
    # 5. Evaluate model
    print("\n5. Evaluating model performance...")
    
    # Training set accuracy
    train_accuracy = network.accuracy(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # Test set accuracy
    test_accuracy = network.accuracy(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # # 6. Visualize training process
    # print("\n6. Plotting training loss curves...")
    # plt.figure(figsize=(12, 4))
    
    # # Training loss
    # plt.subplot(1, 2, 1)
    # plt.plot(losses, 'b-', linewidth=2)
    # plt.title('Training Loss Curve')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.grid(True, alpha=0.3)
    
    # # Loss change rate
    # plt.subplot(1, 2, 2)
    # loss_changes = np.diff(losses)
    # plt.plot(loss_changes, 'r-', linewidth=2)
    # plt.title('Loss Change Rate')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss Change')
    # plt.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.show()
    
    # 7. Prediction examples
    print("\n7. Prediction examples...")
    
    # Select some test samples for prediction
    num_examples = 16
    indices = np.random.choice(X_test.shape[0], num_examples, replace=False)
    X_examples = X_test[indices]
    y_examples = y_test[indices]
    
    # Make predictions
    predictions = network.predict_classes(X_examples)
    true_labels = np.argmax(y_examples, axis=1) if len(y_examples.shape) > 1 else y_examples
    
    # # Visualize prediction results
    # fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    # axes = axes.ravel()
    
    # for i in range(num_examples):
    #     # Display image
    #     img = X_examples[i].reshape(28, 28)
    #     axes[i].imshow(img, cmap='gray')
        
    #     # Display prediction results
    #     pred_label = predictions[i]
    #     true_label = true_labels[i]
    #     color = 'green' if pred_label == true_label else 'red'
        
    #     axes[i].set_title(f'Pred: {pred_label}\nTrue: {true_label}', 
    #                      color=color, fontsize=12)
    #     axes[i].axis('off')
    
    # plt.suptitle('Prediction Examples (Green=Correct, Red=Wrong)', fontsize=16)
    # plt.tight_layout()
    # plt.show()
    
    # 8. Detailed performance analysis
    print("\n8. Detailed performance analysis...")
    
    # Calculate accuracy for each digit
    all_predictions = network.predict_classes(X_test)
    all_true_labels = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    print("Accuracy by class:")
    for digit in range(10):
        digit_mask = all_true_labels == digit
        if np.sum(digit_mask) > 0:
            digit_accuracy = np.mean(all_predictions[digit_mask] == digit)
            digit_count = np.sum(digit_mask)
            print(f"  Digit {digit}: {digit_accuracy:.4f} ({digit_accuracy*100:.1f}%) - {digit_count} samples")
    
    # Confusion matrix
    print("\nConfusion matrix (first 5x5):")
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(all_predictions)):
        confusion_matrix[all_true_labels[i], all_predictions[i]] += 1
    
    # # Display first 5x5 confusion matrix
    # for i in range(5):
    #     row = confusion_matrix[i][:5]
    #     print(f"  True {i}: {row}")
    
    # 9. Save the trained model
    print("\n9. Saving the trained model...")
    save_model_info(network, model_dir='./saved_models')
if __name__ == "__main__":
    main()
