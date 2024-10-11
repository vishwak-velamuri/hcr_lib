import tensorflow as tf
from keras import layers, models
import numpy as np
import networkx as nx
from typing import List

class KANModel:
    def __init__(self, input_dim: int, hidden_layers: List[int] = [128, 64], output_dim: int = 10,
                 activation: str = 'relu', final_activation: str = 'sigmoid', l2_reg: float = 0.01):
        """
        Initializes the KAN-like neural network model with a graph-based transformation.
        
        Args:
            input_dim (int): Number of input features.
            hidden_layers (List[int]): List specifying the number of units in each hidden layer.
            output_dim (int): Number of output units (1 for binary classification).
            activation (str): Activation function for hidden layers.
            final_activation (str): Activation function for output layer.
            l2_reg (float): L2 regularization strength.
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = activation
        self.final_activation = final_activation
        self.l2_reg = l2_reg
        self.transformed_input_dim = input_dim * 5  # Update this based on the actual transformation
        self.model = self._build_model()

    def _build_model(self) -> models.Model:
        model = models.Sequential()
        
        # Input layer (using the transformed input dimension)
        model.add(layers.InputLayer(shape=(self.transformed_input_dim,)))

        # Add hidden layers with L2 regularization and dropout
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation=self.activation, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)))
            model.add(layers.Dropout(0.2))  # Add dropout for regularization
        
        # Output layer
        model.add(layers.Dense(self.output_dim, activation=self.final_activation))
        
        # Use a custom learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)
        
        # Compile the model with the scheduled learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def build_graph(self, input_data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Builds a simple adjacency matrix graph based on input data correlations.
        
        Args:
            input_data (np.ndarray): Input data for building the graph (features).
            threshold (float): Correlation threshold for edge creation.
        
        Returns:
            np.ndarray: Adjacency matrix representing the graph.
        """
        num_features = input_data.shape[1]
        G = nx.Graph()
        
        # Add nodes representing features
        for i in range(num_features):
            G.add_node(i)
        
        # Build edges with correlation threshold
        corr_matrix = np.corrcoef(input_data.T)  # Correlation between features
        
        for i in range(num_features):
            for j in range(i + 1, num_features):
                if abs(corr_matrix[i, j]) > threshold:  # Add edge if correlation is high
                    G.add_edge(i, j, weight=corr_matrix[i, j])
        
        adjacency_matrix = nx.adjacency_matrix(G).todense()
        return adjacency_matrix

    def kan_transform(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs KAN-like transformation using a graph adjacency matrix to enhance input data.
        
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
            input_data,  # Include original features
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

        # Model checkpointing to save the best model
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras', monitor='val_loss', save_best_only=True)

        # Train the model with early stopping and checkpointing
        history = self.model.fit(
            X_transformed, y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_split=0.2,
            callbacks=[early_stopping, model_checkpoint]
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
        self.model.save(filepath, save_format='keras')

    def load_model(self, filepath: str) -> None:
        """
        Loads a trained model from a file.
        
        Args:
            filepath (str): Path where the model is saved.
        """
        self.model = models.load_model(filepath, compile=True)

# Example usage
# Example usage
if __name__ == '__main__':
    import numpy as np
    from utils import load_mnist, preprocess_data, one_hot_encode, accuracy_score
    from visualization import plot_training_history, plot_comparison
    
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, x_test = preprocess_data(x_train, x_test)
    y_train_onehot = one_hot_encode(y_train)
    y_test_onehot = one_hot_encode(y_test)

    # Create the KANModel instance
    input_dim = x_train.shape[1]
    num_classes = 10  # MNIST has 10 classes
    kan_model = KANModel(input_dim=input_dim, hidden_layers=[128, 64], output_dim=num_classes, final_activation='softmax', l2_reg=0.001)

    # Train the model
    print("Training the model...")
    history = kan_model.train(x_train, y_train_onehot, epochs=20, batch_size=128)

    # Plot training history
    plot_training_history(history.history)

    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = kan_model.evaluate(x_test, y_test_onehot)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Make predictions
    print("Making predictions...")
    predictions = kan_model.predict(x_test[:1000])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test_onehot[:1000], axis=1)

    # Calculate and print accuracy
    acc = accuracy_score(true_classes, predicted_classes)
    print(f"Accuracy on 1000 test samples: {acc}")

    # Plot comparison (for the first 100 samples)
    plot_comparison(range(100), true_classes[:100], predicted_classes[:100], "MNIST Predictions")

    # Save and load the model
    print("Saving the model...")
    kan_model.save_model('kan_model.keras')
    
    print("Loading the model...")
    kan_model.load_model('kan_model.keras')
    
    # Re-evaluate after loading to ensure it works
    print("Re-evaluating the loaded model...")
    loss, accuracy = kan_model.evaluate(x_test, y_test_onehot)
    print(f"Re-loaded Model - Test Loss: {loss}, Test Accuracy: {accuracy}")