
import os
import numpy as np
import pandas as pd
import tempfile
import unittest

from the_wizz import pair_maker, pair_collapser


class TestPairCollapser(unittest.TestCase):

    def setUp(self):

        # Seed all random numbers for reproducibility.
        np.random.seed(1234)

        # Create a random catalog centered at the pole with a redshift
        # distribution that looks kind of like a mag limited sample.
        self.n_objects = 1000
        decs = np.degrees(
            np.pi / 2 - np.arccos(np.random.uniform(np.cos(np.radians(1.0)),
                                                    np.cos(0),
                                                    size=self.n_objects)))
        ras = np.random.uniform(0, 360, size=self.n_objects)
        redshifts = np.random.lognormal(mean=-1,
                                        sigma=0.5,
                                        size=self.n_objects)
        ids = np.arange(self.n_objects)
        self.catalog = {"id": ids,
                        "ra": ras,
                        "dec": decs,
                        "redshift": redshifts}

        self.z_min = 0.05
        self.z_max = 3.0

        self.r_mins = [0.1, 1]
        self.r_maxes = [1, 10]
        self.r_min = np.min(self.r_mins)
        self.r_max = np.max(self.r_maxes)

        self.expected_columns = ["ref_id",
                                 "redshift"]
        for r_min, r_max in zip(self.r_mins, self.r_maxes):
            self.expected_columns.append("Mpc%.2ft%.2f_counts" %
                                         (r_min, r_max))
            self.expected_columns.append("Mpc%.2ft%.2f_weights" %
                                         (r_min, r_max))
        self.output_path = tempfile.mkdtemp(
            dir=os.path.dirname(__file__))

    def test_collapse_pairs(self):
        pass

    def test_collapse_pairs_ref_id(self):
        pass

    def test_find_trim_indexes(self):
        """Test that arrays are trimmed correctly. 
        """
        # Test a fully contained array
        test_ids = np.arange(10, 20, dtype=np.int)
        input_ids = np.arange(15, 18, dtype=np.int)
        (start_idx, end_idx) = pair_collapser.find_trim_indexes(input_ids,
                                                                test_ids)
        self.assertEqual(start_idx, 5)
        self.assertEqual(end_idx, 8)

        (start_idx, end_idx) = pair_collapser.find_trim_indexes(test_ids,
                                                                input_ids)
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 3)

        # Test overlapping arrays.
        input_ids = np.arange(15, dtype=np.int)
        (start_idx, end_idx) = pair_collapser.find_trim_indexes(input_ids,
                                                                test_ids)
        self.assertEqual(start_idx, 0)
        self.assertEqual(end_idx, 5)

        (start_idx, end_idx) = pair_collapser.find_trim_indexes(test_ids,
                                                                input_ids)
        self.assertEqual(start_idx, 10)
        self.assertEqual(end_idx, 15)

        # Test distinct arrays
        input_ids = np.arange(8, dtype=np.int)
        (start_idx, end_idx) = pair_collapser.find_trim_indexes(input_ids,
                                                                test_ids)
        self.assertEqual(start_idx, end_idx)

        (start_idx, end_idx) = pair_collapser.find_trim_indexes(test_ids,
                                                                input_ids)
        self.assertEqual(start_idx, end_idx)

    def test_find_pairs(self):
        """Test that finding pairs and weights works.
        """
        test_ids = np.arange(10, dtype=np.int)
        test_weights = np.arange(10, dtype=np.float)

        input_ids = np.array([2, 3, 5, 11])
        input_weights = np.array([4, 6, 10, 22])

        input_weights, matched_weights = pair_collapser.find_pairs(
            input_ids,
            test_ids,
            input_weights,
            test_weights)

        ans_input_weights = np.array([4, 6, 10])
        ans_test_weights = np.array([2, 3, 5])

        for in_w, m_w, a_in_w, a_m_w in zip(input_weights,
                                            matched_weights,
                                            ans_input_weights,
                                            ans_test_weights):
            self.assertEqual(in_w, a_in_w)
            self.assertEqual(m_w, a_m_w)


if __name__ == "__main__":

    unittest.main()
