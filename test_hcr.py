import unittest
import numpy as np
from hcr import HCR

class TestHCR(unittest.TestCase):

    def setUp(self):
        self.hcr = HCR(m=5)
        self.data_1d = np.random.rand(100, 1)
        self.data_nd = np.random.rand(100, 3)

    def test_normalize_gaussian(self):
        normalized_data = self.hcr.normalize_gaussian(self.data_1d)
        self.assertEqual(normalized_data.shape, self.data_1d.shape)
        self.assertTrue(np.all(normalized_data >= 0) and np.all(normalized_data <= 1))

        with self.assertRaises(ValueError):
            self.hcr.normalize_gaussian(np.zeros((100, 1)))

    def test_normalize_edf(self):
        normalized_data = self.hcr.normalize_edf(self.data_1d)
        self.assertEqual(normalized_data.shape, self.data_1d.shape)
        self.assertTrue(np.all(normalized_data >= 0) and np.all(normalized_data <= 1))

        with self.assertRaises(ValueError):
            self.hcr.normalize_edf(np.array([]))

    def test_legendre_basis(self):
        basis = self.hcr.legendre_basis(5, self.data_1d.flatten())
        self.assertEqual(basis.shape, (6, 100))

        with self.assertRaises(ValueError):
            self.hcr.legendre_basis(5, np.array([]))

    def test_estimate_coefficients(self):
        hcr_nd = HCR(m=5, data=self.data_nd)
        coeffs = hcr_nd.estimate_coefficients(self.data_nd)
        self.assertEqual(coeffs.shape, ((5 + 1) ** 3,))

        with self.assertRaises(ValueError):
            hcr_nd.estimate_coefficients(np.empty((0, 3)))

        with self.assertRaises(ValueError):
            hcr_nd.estimate_coefficients(np.random.rand(100, 2))

    def test_density(self):
        hcr_nd = HCR(m=5, data=self.data_nd)
        hcr_nd.estimate_coefficients(self.data_nd)
        x = np.random.rand(1, self.data_nd.shape[1])
        density = hcr_nd.density(x)
        self.assertIsInstance(density, float)
        self.assertGreaterEqual(density, 0)

    def test_conditional_density(self):
        hcr_nd = HCR(m=5, data=self.data_nd)
        hcr_nd.estimate_coefficients(self.data_nd)
        x = np.random.rand(1, self.data_nd.shape[1])
        cond_density = hcr_nd.conditional_density(x, [0, 1])
        self.assertIsInstance(cond_density, float)
        self.assertGreaterEqual(cond_density, 0)

    def test_marginal_density(self):
        hcr_nd = HCR(m=5, data=self.data_nd)
        hcr_nd.estimate_coefficients(self.data_nd)
        x = np.random.rand(1, self.data_nd.shape[1])
        marg_density = hcr_nd.marginal_density(x, [0, 1])
        self.assertIsInstance(marg_density, float)
        self.assertGreaterEqual(marg_density, 0)

    def test_cross_validate(self):
        hcr_nd = HCR(m=5, data=self.data_nd)
        log_likelihood = hcr_nd.cross_validate(self.data_nd)
        self.assertIsInstance(log_likelihood, float)

    def test_predict_conditional_distribution(self):
        x = self.data_nd[:, :2]
        y = self.data_nd[:, 2]
        hcr_nd = HCR(m=5, data=self.data_nd)
        cond_dist_func = hcr_nd.predict_conditional_distribution(x, y)
        x_new = np.random.rand(1, 2)
        cond_dist = cond_dist_func(x_new)
        y_test = np.random.rand()
        density = cond_dist(y_test)
        self.assertIsInstance(density, float)
        self.assertGreaterEqual(density, 0)

    def test_analyze_correlations(self):
        hcr_nd = HCR(m=5, data=self.data_nd)
        components, correlations = hcr_nd.analyze_correlations(self.data_nd)
        self.assertEqual(components.shape, (2, 3))
        self.assertEqual(correlations.shape, (2, 3))

    def test_calculate_conditional_entropy(self):
        hcr_nd = HCR(m=5, data=self.data_nd)
        relevance, novelty = hcr_nd.calculate_conditional_entropy(self.data_nd, 0)
        self.assertIsInstance(relevance, float)
        self.assertIsInstance(novelty, float)
        self.assertGreaterEqual(relevance, 0)
        self.assertGreaterEqual(novelty, 0)

    def test_time_dependent_modeling(self):
        hcr_td = HCR(m=5, data=self.data_nd, time_dependent=True, alpha=0.1)
        coeffs_td = hcr_td.estimate_coefficients(self.data_nd)
        self.assertEqual(coeffs_td.shape, ((5 + 1) ** 3,))

        # Compare with non-time-dependent model
        hcr_non_td = HCR(m=5, data=self.data_nd)
        coeffs_non_td = hcr_non_td.estimate_coefficients(self.data_nd)

        # Check that coefficients are different
        self.assertFalse(np.allclose(coeffs_td, coeffs_non_td))

    def test_calibration(self):
        hcr_nd = HCR(m=5, data=self.data_nd)
        hcr_nd.estimate_coefficients(self.data_nd)
        x = np.random.rand(1, self.data_nd.shape[1])
        density = hcr_nd.density(x)
        
        # Check that density is non-negative
        self.assertGreaterEqual(density, 0)

        # Check that very small densities are calibrated to a minimum value
        hcr_nd.coefficients = np.zeros_like(hcr_nd.coefficients)  # Force density to be zero
        calibrated_density = hcr_nd.density(x)
        self.assertGreater(calibrated_density, 0)

    def test_higher_order_dependencies(self):
        # Generate data with known higher-order dependencies
        x = np.linspace(0, 1, 100)
        y = x**2
        z = x**3 + y**2
        data = np.column_stack((x, y, z))

        hcr_high_order = HCR(m=5, data=data)
        coeffs = hcr_high_order.estimate_coefficients(data)

        # Check that higher-order coefficients are non-zero
        higher_order_coeffs = coeffs[-(5+1)**3:]
        self.assertTrue(np.any(np.abs(higher_order_coeffs) > 1e-6))

    def test_pca_correlation_analysis(self):
        hcr_nd = HCR(m=5, data=self.data_nd)
        components, correlations = hcr_nd.analyze_correlations(self.data_nd, n_components=2)

        # Check shapes
        self.assertEqual(components.shape, (2, 3))
        self.assertEqual(correlations.shape, (2, 3))

        # Check that correlations are between -1 and 1
        self.assertTrue(np.all(correlations >= -1) and np.all(correlations <= 1))

    def test_conditional_entropy_analysis(self):
        hcr_nd = HCR(m=5, data=self.data_nd)
        
        for i in range(self.data_nd.shape[1]):
            relevance, novelty = hcr_nd.calculate_conditional_entropy(self.data_nd, i)
            
            # Check that relevance and novelty are non-negative
            self.assertGreaterEqual(relevance, 0)
            self.assertGreaterEqual(novelty, 0)

            # Check that relevance is not greater than total entropy
            total_entropy = hcr_nd._calculate_entropy(self.data_nd)
            self.assertLessEqual(relevance, total_entropy)

    def test_cross_validation_model_selection(self):
        # Test cross-validation for different model orders
        max_order = 7
        cv_scores = []

        for m in range(1, max_order + 1):
            hcr = HCR(m=m, data=self.data_nd)
            cv_score = hcr.cross_validate(self.data_nd)
            cv_scores.append(cv_score)

        # Check that we have the expected number of scores
        self.assertEqual(len(cv_scores), max_order)

        # Check that scores are finite
        self.assertTrue(all(np.isfinite(score) for score in cv_scores))

if __name__ == "__main__":
    unittest.main()