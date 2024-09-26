import unittest
import torch
import numpy as np
from nn import HCRKAN, HCRNN, HCRLayer, train_hcrkan, direct_estimation_training, tensor_decomposition_training, information_bottleneck_training

class TestNN(unittest.TestCase):
    def setUp(self):
        self.input_dim = 5
        self.hidden_layers = [10, 5]
        self.output_dim = 1
        self.m = 3
        self.model = HCRKAN(self.input_dim, self.hidden_layers, self.output_dim, self.m)
        self.X = np.random.rand(100, self.input_dim)
        self.y = np.sin(self.X[:, 0] * np.pi) + 0.1 * np.random.randn(100)

    def test_hcr_layer(self):
        layer = HCRLayer(self.input_dim, self.output_dim, self.m)
        x = torch.rand(10, self.input_dim)
        output = layer(x)
        self.assertEqual(output.shape, (10, self.output_dim))

    def test_hcrnn(self):
        hcrnn = HCRNN([self.input_dim] + self.hidden_layers + [self.output_dim], self.m)
        x = torch.rand(10, self.input_dim)
        output = hcrnn(x)
        self.assertEqual(output.shape, (10, self.output_dim))

    def test_hcrkan(self):
        x = torch.tensor(self.X).float()
        output = self.model(x)
        self.assertEqual(output.shape, (100, self.output_dim * 2))

    def test_train_hcrkan(self):
        losses = train_hcrkan(self.model, self.X, self.y, epochs=10)
        self.assertEqual(len(losses), 10)

    def test_direct_estimation_training(self):
        direct_estimation_training(self.model, self.X, self.y, self.m)
        x = torch.tensor(self.X).float()
        output = self.model(x)
        self.assertEqual(output.shape, (100, self.output_dim * 2))

    def test_tensor_decomposition_training(self):
        tensor_decomposition_training(self.model, self.X, self.y, max_iter=10)
        x = torch.tensor(self.X).float()
        output = self.model(x)
        self.assertEqual(output.shape, (100, self.output_dim * 2))

    def test_information_bottleneck_training(self):
        information_bottleneck_training(self.model, self.X, self.y, beta=1.0, max_iter=10)
        x = torch.tensor(self.X).float()
        output = self.model(x)
        self.assertEqual(output.shape, (100, self.output_dim * 2))

if __name__ == '__main__':
    unittest.main()