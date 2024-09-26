import unittest
import numpy as np
from kan import KANModel

class TestKANModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.kan_model = KANModel(input_dim=self.input_dim)
        self.X_train = np.random.rand(100, self.input_dim)
        self.y_train = np.random.randint(0, 2, size=(100,))

    def test_build_graph(self):
        adj_matrix = self.kan_model.build_graph(self.X_train)
        self.assertEqual(adj_matrix.shape, (self.input_dim, self.input_dim))

    def test_kan_transform(self):
        transformed_data = self.kan_model.kan_transform(self.X_train)
        self.assertEqual(transformed_data.shape, self.X_train.shape)

    def test_train(self):
        history = self.kan_model.train(self.X_train, self.y_train, epochs=2)
        self.assertIsInstance(history, object)

    def test_evaluate(self):
        self.kan_model.train(self.X_train, self.y_train, epochs=2)
        X_test = np.random.rand(20, self.input_dim)
        y_test = np.random.randint(0, 2, size=(20,))
        evaluation = self.kan_model.evaluate(X_test, y_test)
        self.assertEqual(len(evaluation), 2)

    def test_predict(self):
        self.kan_model.train(self.X_train, self.y_train, epochs=2)
        X_input = np.random.rand(5, self.input_dim)
        predictions = self.kan_model.predict(X_input)
        self.assertEqual(predictions.shape, (5, 1))

if __name__ == '__main__':
    unittest.main()