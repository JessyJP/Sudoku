import unittest
import numpy as np
from sudoku3d.indexing_3d import (coordinates_to_vectors_3d, get_linear_element_indices_of_non_zero_3d,
                                   get_linear_coordinates_3d, get_random_element_indices_of_non_zero_3d,
                                   get_zigzag_matrix_indices_3d, get_block_indices_3d, get_diagonal_block_indices_3d)

class TestIndexing3D(unittest.TestCase):

    def test_coordinates_to_vectors_3d(self):
        # Test for coordinates_to_vectors_3d method: Converting coordinate list to separate x, y, z vectors.
        test_cases = [
            ([(1, 2, 3), (4, 5, 6)], ([1, 4], [2, 5], [3, 6])),
            ([(0, 0, 0), (1, 1, 1)], ([0, 1], [0, 1], [0, 1])),
            ([(2, 4, 6), (3, 6, 9), (8, 10, 12)], ([2, 3, 8], [4, 6, 10], [6, 9, 12]))
        ]
        for inputs, expected in test_cases:
            with self.subTest(inputs=inputs):
                result = coordinates_to_vectors_3d(inputs)
                self.assertEqual(result, expected)

    def test_get_linear_element_indices_of_non_zero_3d(self):
        # Test for get_linear_element_indices_of_non_zero_3d method: Extracting non-zero element indices from a 3D numpy array.
        test_cases = [
            (np.array([[[0, 1], [0, 0]], [[1, 0], [0, 2]]]), ((0, 1, 1), (0, 0, 1), (1, 1))),
            (np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), ((0, 0, 0, 0, 1, 1, 1, 1), (0, 0, 1, 1, 0, 0, 1, 1), (0, 1, 0, 1, 0, 1, 0, 1))),
            (np.zeros((2, 2, 2)), (tuple(), tuple(), tuple()))
        ]
        for B, expected in test_cases:
            with self.subTest(B=B):
                result = get_linear_element_indices_of_non_zero_3d(B)
                self.assertEqual(result, expected)

    def test_get_linear_coordinates_3d(self):
        # Test for get_linear_coordinates_3d method: Generating linear coordinates for a cubical grid with different orderings.
        test_cases = [
            (2, 'xyz', [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]),
            (2, 'xzy', [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]),
            (3, 'yxz', [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0), (1, 2, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0), (0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1), (1, 1, 1), (1, 2, 1), (2, 0, 1), (2, 1, 1), (2, 2, 1), (0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 0, 2), (1, 1, 2), (1, 2, 2), (2, 0, 2), (2, 1, 2), (2, 2, 2)])
        ]
        for N, order, expected in test_cases:
            with self.subTest(N=N, order=order):
                result = get_linear_coordinates_3d(N, order)
                self.assertEqual(result, expected)

    def test_get_random_element_indices_of_non_zero_3d(self):
        # Test for get_random_element_indices_of_non_zero_3d method: Validating shape and uniqueness for randomized index shuffling.
        B = np.array([[[0, 1], [0, 0]], [[1, 0], [0, 2]]])
        X, Y, Z = get_random_element_indices_of_non_zero_3d(B)
        self.assertEqual(len(X), 3)
        self.assertEqual(len(set(zip(X, Y, Z))), 3)

    def test_get_zigzag_matrix_indices_3d(self):
        # Test for get_zigzag_matrix_indices_3d method: Checking count and uniqueness of zigzag traversal indices in a 3D matrix.
        n = 2
        expected_length = n * n * n
        result = get_zigzag_matrix_indices_3d(n)
        self.assertEqual(len(result), expected_length)
        self.assertEqual(len(set(result)), expected_length)

    def test_get_block_indices_3d(self):
        # Test for get_block_indices_3d method: Confirming block index calculation within a 3D grid.
        test_cases = [
            (0, 0, 0, 2, [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]),
            (1, 1, 1, 2, [(2, 2, 2), (2, 2, 3), (2, 3, 2), (2, 3, 3), (3, 2, 2), (3, 2, 3), (3, 3, 2), (3, 3, 3)]),
            (0, 0, 1, 1, [(0, 0, 1)])
        ]
        for block_x, block_y, block_z, g, expected in test_cases:
            with self.subTest(block_x=block_x, block_y=block_y, block_z=block_z, g=g):
                result = get_block_indices_3d(block_x, block_y, block_z, g)
                self.assertEqual(result, expected)

    def test_get_diagonal_block_indices_3d(self):
        # Test for get_diagonal_block_indices_3d method: Checking diagonal block indices in a cubic grid.
        test_cases = [
            (2, False, [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]),
            (2, True, [(1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1), (0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1)]),
            (1, False, [(0, 0, 0)])
        ]
        for g, reverse, expected in test_cases:
            with self.subTest(g=g, reverse=reverse):
                result = get_diagonal_block_indices_3d(g, reverse)
                self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
