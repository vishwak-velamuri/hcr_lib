import unittest
import numpy as np
from hcr import HCR

class TestHCR(unittest.TestCase):
    def setUp(self):
        self.hcr = HCR(m=5)
        self.data_1d = np.random.randn(1000)
        self.data_nd = np.random.randn(1000, 3)

    def test_normalize_gaussian(self):
        normalized = HCR.normalize_gaussian(self.data_1d)
        self.assertTrue(np.all((normalized >= 0) & (normalized <= 1)))

    def test_normalize_edf(self):
        normalized = HCR.normalize_edf(self.data_1d)
        self.assertTrue(np.all((normalized >= 0) & (normalized <= 1)))

    def test_legendre_basis(self):
        x = np.linspace(0, 1, 100)
        basis = HCR.legendre_basis(5, x)
        self.assertEqual(basis.shape, (6, 100))

    def test_estimate_coefficients_1d(self):
        coeffs = self.hcr.estimate_coefficients_1d(self.data_1d)
        self.assertEqual(coeffs.shape, (6,))

    def test_estimate_coefficients_nd(self):
        coeffs = self.hcr.estimate_coefficients_nd(self.data_nd)
        expected_size = (self.hcr.m + 1) ** self.data_nd.shape[1]
        self.assertEqual(coeffs.shape, (expected_size,))

    def test_density_1d(self):
        self.hcr.estimate_coefficients_1d(self.data_1d)
        x = np.linspace(0, 1, 100)
        density = self.hcr.density_1d(x)
        self.assertEqual(density.shape, (100,))

    def test_density_nd(self):
        self.hcr.estimate_coefficients_nd(self.data_nd)
        x = np.random.rand(1, self.data_nd.shape[1])  # Create a single point with the correct dimensionality
        density = self.hcr.density_nd(x)
        self.assertTrue(isinstance(density, float))

if __name__ == '__main__':
    unittest.main()