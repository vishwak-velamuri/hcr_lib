import unittest
import numpy as np
from kan import KANModel

class TestKANModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.kan_model = KANModel(input_dim=self.input_dim)
        # Generate more realistic data
        np.random.seed(42)  # For reproducibility
        self.X_train = np.random.randn(1000, self.input_dim)  # More samples
        self.y_train = (np.sum(self.X_train[:, :5], axis=1) > 0).astype(int)  # Create a pattern

    def test_build_graph(self):
        adj_matrix = self.kan_model.build_graph(self.X_train)
        self.assertEqual(adj_matrix.shape, (self.input_dim, self.input_dim))

    def test_kan_transform(self):
        transformed_data = self.kan_model.kan_transform(self.X_train)
        self.assertEqual(transformed_data.shape[0], self.X_train.shape[0])
        self.assertGreater(transformed_data.shape[1], self.X_train.shape[1])  # Due to added features

    def test_train(self):
        history = self.kan_model.train(self.X_train, self.y_train, epochs=20, batch_size=64)
        self.assertIsInstance(history, object)
        # Check if accuracy has improved
        self.assertGreater(history.history['accuracy'][-1], 0.6)

    def test_evaluate(self):
        self.kan_model.train(self.X_train, self.y_train, epochs=20, batch_size=64)
        X_test = np.random.randn(200, self.input_dim)
        y_test = (np.sum(X_test[:, :5], axis=1) > 0).astype(int)
        loss, accuracy = self.kan_model.evaluate(X_test, y_test)
        self.assertGreater(accuracy, 0.6)

    def test_predict(self):
        self.kan_model.train(self.X_train, self.y_train, epochs=20, batch_size=64)
        X_input = np.random.randn(5, self.input_dim)
        predictions = self.kan_model.predict(X_input)
        self.assertEqual(predictions.shape, (5, 1))
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))

if __name__ == '__main__':
    unittest.main()