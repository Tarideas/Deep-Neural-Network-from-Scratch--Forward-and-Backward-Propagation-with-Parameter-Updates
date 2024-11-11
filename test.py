# test.py

import unittest
import numpy as np
from main import initialize_parameters, linear_forward, sigmoid, relu, linear_activation_forward, L_model_forward, compute_cost, sigmoid_backward, relu_backward, linear_backward, linear_activation_backward, L_model_backward, update_parameters

class TestNeuralNetwork(unittest.TestCase):
    
    def test_initialize_parameters(self):
        layer_dims = [3, 2, 1]
        parameters = initialize_parameters(layer_dims)
        self.assertEqual(parameters['W1'].shape, (2, 3))
        self.assertEqual(parameters['b1'].shape, (2, 1))
        self.assertEqual(parameters['W2'].shape, (1, 2))
        self.assertEqual(parameters['b2'].shape, (1, 1))
    
    def test_linear_forward(self):
        A = np.array([[1, 2], [3, 4]])
        W = np.array([[1, 0], [0, 1]])
        b = np.array([[1], [1]])
        Z, _ = linear_forward(A, W, b)
        expected_Z = np.array([[3, 4], [5, 6]])
        np.testing.assert_array_equal(Z, expected_Z)
    
    def test_sigmoid(self):
        Z = np.array([[0, 2]])
        A, _ = sigmoid(Z)
        expected_A = np.array([[0.5, 0.88079708]])
        np.testing.assert_array_almost_equal(A, expected_A)

    def test_relu(self):
        Z = np.array([[-1, 2]])
        A, _ = relu(Z)
        expected_A = np.array([[0, 2]])
        np.testing.assert_array_equal(A, expected_A)

    def test_linear_activation_forward(self):
        A_prev = np.array([[1, 2], [3, 4]])
        W = np.array([[1, 0], [0, 1]])
        b = np.array([[1], [1]])
        A, _ = linear_activation_forward(A_prev, W, b, activation="relu")
        expected_A = np.array([[3, 4], [5, 6]])
        np.testing.assert_array_equal(A, expected_A)

    def test_compute_cost(self):
        AL = np.array([[0.8, 0.9]])
        Y = np.array([[1, 0]])
        cost = compute_cost(AL, Y)
        expected_cost = 0.21616187468057912
        self.assertAlmostEqual(cost, expected_cost, places=7)

    def test_update_parameters(self):
        parameters = {
            "W1": np.array([[1, 2], [3, 4]]),
            "b1": np.array([[1], [2]]),
            "W2": np.array([[5, 6]]),
            "b2": np.array([[1]])
        }
        grads = {
            "dW1": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "db1": np.array([[0.1], [0.2]]),
            "dW2": np.array([[0.5, 0.6]]),
            "db2": np.array([[0.1]])
        }
        learning_rate = 0.1
        updated_params = update_parameters(parameters, grads, learning_rate)
        expected_W1 = np.array([[0.99, 1.98], [2.97, 3.96]])
        expected_b1 = np.array([[0.99], [1.98]])
        expected_W2 = np.array([[4.95, 5.94]])
        expected_b2 = np.array([[0.99]])
        
        np.testing.assert_array_almost_equal(updated_params["W1"], expected_W1)
        np.testing.assert_array_almost_equal(updated_params["b1"], expected_b1)
        np.testing.assert_array_almost_equal(updated_params["W2"], expected_W2)
        np.testing.assert_array_almost_equal(updated_params["b2"], expected_b2)

if __name__ == "__main__":
    unittest.main()
