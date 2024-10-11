import matplotlib.pyplot as plt
import numpy as np

def plot_hcr_density(hcr, x_range=(0, 1), num_points=1000):
    """
    Plot the HCR density function.
    
    Args:
        hcr (HCR): Trained HCR object
        x_range (tuple): Range of x values to plot
        num_points (int): Number of points to use for plotting
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = hcr.density_1d(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title("HCR Density Function")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

def plot_training_history(history):
    """
    Plot the training history of a model.
    
    Args:
        history (dict): Training history containing 'loss' and 'accuracy' keys
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_comparison(x, y_true, y_pred, title):
    """
    Plot a comparison between true and predicted values.
    
    Args:
        x (np.ndarray): Input features
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_true, label='True', alpha=0.5)
    plt.scatter(x, y_pred, label='Predicted', alpha=0.5)
    plt.title(title)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.show()