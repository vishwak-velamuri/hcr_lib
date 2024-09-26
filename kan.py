import tensorflow as tf
from keras import layers, models
import numpy as np
import networkx as nx
from typing import List

class KANModel:
    def __init__(self, input_dim: int, hidden_layers: List[int] = [128, 64], output_dim: int = 1, activation: str = 'relu', final_activation: str = 'sigmoid'):
        """
        Initializes the KAN-like neural network model with a graph-based transformation.
        
        Args:
            input_dim (int): Number of input features.
            hidden_layers (List[int]): List specifying the number of units in each hidden layer.
            output_dim (int): Number of output units (1 for binary classification).
            activation (str): Activation function for hidden layers.
            final_activation (str): Activation function for output layer.
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = activation
        self.final_activation = final_activation
        self.model = self._build_model()

    def _build_model(self) -> models.Model:
        model = models.Sequential()
        
        # Input layer
        model.add(layers.InputLayer(shape=(self.input_dim,)))

        # Add hidden layers with L2 regularization and dropout
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation=self.activation, kernel_regularizer=regularizers.l2(0.01)))
            model.add(layers.Dropout(0.2))  # Add dropout for regularization
        
        # Output layer
        model.add(layers.Dense(self.output_dim, activation=self.final_activation))
        
        # Use a custom learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)
        
        # Compile the model with the scheduled learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def build_graph(self, input_data: np.ndarray) -> np.ndarray:
        """
        Builds a simple adjacency matrix graph based on input data correlations.
        Each node is a feature, and edges represent correlation strength.
        
        Args:
            input_data (np.ndarray): Input data for building the graph (features).
        
        Returns:
            np.ndarray: Adjacency matrix representing the graph.
        """
        num_features = input_data.shape[1]
        G = nx.Graph()
        
        # Add nodes representing features
        for i in range(num_features):
            G.add_node(i)
        
        # Build edges with a simple correlation threshold (can be replaced with knowledge-based relations)
        corr_matrix = np.corrcoef(input_data.T)  # Correlation between features
        threshold = 0.5  # Adjust based on domain knowledge
        
        for i in range(num_features):
            for j in range(i + 1, num_features):
                if abs(corr_matrix[i, j]) > threshold:  # Add edge if correlation is high
                    G.add_edge(i, j, weight=corr_matrix[i, j])
        
        adjacency_matrix = nx.adjacency_matrix(G).todense()
        return adjacency_matrix

    def kan_transform(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs KAN-like transformation using a graph adjacency matrix to enhance input data.
        Incorporates relationships between input features.
        
        Args:
            input_data (np.ndarray): Input data (e.g., features) to be transformed.
        
        Returns:
            np.ndarray: Transformed input data, which includes graph-enhanced features.
        """
        adjacency_matrix = self.build_graph(input_data)
        
        # Combine input data with adjacency matrix (feature interactions)
        transformed_data = np.dot(input_data, adjacency_matrix)
        
        # Add non-linear transformations
        transformed_data = np.column_stack([
            transformed_data,
            np.sin(transformed_data),
            np.cos(transformed_data),
            np.tanh(transformed_data)
        ])
        
        # Normalize transformed data
        epsilon = 1e-8
        norm = np.linalg.norm(transformed_data, axis=1, keepdims=True)
        transformed_data = transformed_data / (norm + epsilon)
        
        return transformed_data

    def train(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int = 32, epochs: int = 10) -> tf.keras.callbacks.History:
        """
        Trains the model on the provided dataset.
        
        Args:
            X_train (np.ndarray): Training input features.
            y_train (np.ndarray): Training labels.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs for training.
        
        Returns:
            tf.keras.callbacks.History: History object from the training process.
        """
        X_transformed = self.kan_transform(X_train)
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train the model with more epochs and early stopping
        history = self.model.fit(
            X_transformed, y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_split=validation_split,
            callbacks=[early_stopping]
        )
        
        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> List[float]:
        """
        Evaluates the model on the test dataset.
        
        Args:
            X_test (np.ndarray): Test input features.
            y_test (np.ndarray): Test labels.
        
        Returns:
            List[float]: Evaluation result (loss and accuracy).
        """
        X_transformed = self.kan_transform(X_test)
        return self.model.evaluate(X_transformed, y_test)

    def predict(self, X_input: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data.
        
        Args:
            X_input (np.ndarray): Input data for prediction.
        
        Returns:
            np.ndarray: Predictions made by the model.
        """
        X_transformed = self.kan_transform(X_input)
        return self.model.predict(X_transformed)

    def save_model(self, filepath: str) -> None:
        """
        Saves the trained model to a file.
        
        Args:
            filepath (str): Path where the model should be saved.
        """
        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """
        Loads a trained model from a file.
        
        Args:
            filepath (str): Path where the model is saved.
        """
        self.model = models.load_model(filepath)


# Example usage
if __name__ == '__main__':
    # Sample data: Replace this with your actual dataset
    X_train = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y_train = np.random.randint(0, 2, size=(1000,))  # Binary target
    
    X_test = np.random.rand(200, 10)  # 200 samples for testing
    y_test = np.random.randint(0, 2, size=(200,))  # Binary target for testing
    
    # Create the KANModel instance
    input_dim = X_train.shape[1]
    kan_model = KANModel(input_dim=input_dim)

    # Train the model
    print("Training the model...")
    kan_model.train(X_train, y_train, epochs=5)

    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = kan_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Predict on new data
    print("Making predictions...")
    predictions = kan_model.predict(X_test)
    print(f"Predictions: {predictions[:5]}")
    
    # Save and load the model
    print("Saving the model...")
    kan_model.save_model('kan_model.h5')
    
    print("Loading the model...")
    kan_model.load_model('kan_model.h5')
    
    # Re-evaluate after loading to ensure it works
    print("Re-evaluating the loaded model...")
    loss, accuracy = kan_model.evaluate(X_test, y_test)
    print(f"Re-loaded Model - Test Loss: {loss}, Test Accuracy: {accuracy}")