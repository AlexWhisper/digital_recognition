#!/usr/bin/env python3
"""
Load and use saved neural network model for prediction
"""

import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNetwork
from mnist_loader import load_mnist_data
import json
import os
import torch

def load_model_info(model_dir=None):
    """
    Load neural network model from saved files using PyTorch
    """
    if model_dir is None:
        # 使用根目录的统一模型加载路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(project_root, 'saved_models', 'mlp_version')
    
    # Load model structure
    with open(os.path.join(model_dir, 'model_structure.json'), 'r') as f:
        model_info = json.load(f)
    
    # Create network with loaded structure
    network = NeuralNetwork(model_info['layer_sizes'], model_info['learning_rate'])
    
    # Load PyTorch model state
    model_path = os.path.join(model_dir, 'model.pth')
    network.load_state_dict(torch.load(model_path))
    network.eval()
    
    print(f"Model loaded from {model_dir}/")
    return network

def predict_single_digit(network, image):
    """
    Predict a single digit image
    """
    # Ensure image is in correct format (1, 784) for batch prediction
    if image.ndim == 1:
        image = image.reshape(1, -1)
    elif image.ndim == 2 and image.shape[0] != 1:
        image = image.reshape(1, -1)
    
    # Get prediction probabilities
    probabilities = network.predict(image)
    predicted_class = network.predict_classes(image)[0]
    confidence = probabilities[0, predicted_class]
    
    return predicted_class, confidence, probabilities[0, :]

def visualize_prediction(network, image, predicted_class, confidence, true_class=None):
    """
    Visualize prediction result
    """
    plt.figure(figsize=(8, 4))
    
    # Show image
    plt.subplot(1, 2, 1)
    img_reshaped = image.reshape(28, 28)
    plt.imshow(img_reshaped, cmap='gray')
    plt.title(f'Input Image')
    plt.axis('off')
    
    # Show prediction results
    plt.subplot(1, 2, 2)
    classes = list(range(10))
    probabilities = network.predict(image.reshape(1, -1))[0, :]
    
    colors = ['red' if i == predicted_class else 'blue' for i in classes]
    bars = plt.bar(classes, probabilities, color=colors, alpha=0.7)
    
    # Highlight predicted class
    bars[predicted_class].set_color('red')
    bars[predicted_class].set_alpha(1.0)
    
    plt.xlabel('Digit Class')
    plt.ylabel('Probability')
    plt.title(f'Prediction: {predicted_class} (Confidence: {confidence:.3f})')
    plt.ylim(0, 1)
    
    # Add true class if provided
    if true_class is not None:
        plt.axvline(x=true_class, color='green', linestyle='--', linewidth=2, 
                   label=f'True: {true_class}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 60)
    print("Loading and Using Saved Neural Network Model")
    print("=" * 60)
    
    # Get the unified model path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(project_root, 'saved_models', 'mlp_version')
    model_structure_path = os.path.join(model_dir, 'model_structure.json')
    
    # Check if saved model exists
    if not os.path.exists(model_structure_path):
        print("No saved model found!")
        print("Please run 'train_and_save.py' first to train and save a model.")
        print(f"Expected model location: {model_dir}")
        return
    
    # Load the saved model
    print("\n1. Loading saved model...")
    network = load_model_info()
    
    # Load some test data
    print("\n2. Loading test data...")
    _, _, X_test, y_test = load_mnist_data(num_train=0, num_test=100)
    
    # Test model performance
    print("\n3. Testing model performance...")
    test_accuracy = network.accuracy(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Make predictions on some examples
    print("\n4. Making predictions on sample images...")
    num_examples = 5
    
    for i in range(num_examples):
        # Get a random test image
        idx = np.random.randint(0, X_test.shape[0])
        image = X_test[idx, :]
        true_class = np.argmax(y_test[idx, :])
        
        # Make prediction
        predicted_class, confidence, probabilities = predict_single_digit(network, image)
        
        print(f"\nExample {i+1}:")
        print(f"  True digit: {true_class}")
        print(f"  Predicted: {predicted_class}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Correct: {'Yes' if predicted_class == true_class else 'No'}")
        
        # Visualize prediction
        visualize_prediction(network, image, predicted_class, confidence, true_class)
        
        # Ask user if they want to continue
        if i < num_examples - 1:
            input("\nPress Enter to see next example...")
    
    # Interactive prediction mode
    print("\n5. Interactive prediction mode...")
    print("You can now test the model with different images.")
    
    while True:
        try:
            # Get a random test image
            idx = np.random.randint(0, X_test.shape[0])
            image = X_test[idx, :]
            true_class = np.argmax(y_test[idx, :])
            
            # Make prediction
            predicted_class, confidence, probabilities = predict_single_digit(network, image)
            
            print(f"\nRandom test image:")
            print(f"  True digit: {true_class}")
            print(f"  Predicted: {predicted_class}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Correct: {'Yes' if predicted_class == true_class else 'No'}")
            
            # Visualize
            visualize_prediction(network, image, predicted_class, confidence, true_class)
            
            # Ask if user wants to continue
            user_input = input("\nPress Enter for another prediction, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
    
    print("\n" + "=" * 60)
    print("Model usage completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
