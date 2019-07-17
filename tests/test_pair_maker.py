
from astropy.cosmology import Planck15
import h5py
import os
import numpy as np
import tempfile
import unittest

from the_wizz import pair_maker


class TestPairMakerUtils(unittest.TestCase):

    def setUp(self):

        # Seed all random numbers for reproducibility.
        np.random.seed(1234)

        # Create a random catalog centered at the pole with a redshift
        # distribution that looks kind of like a mag limited sample.
        self.n_objects = 10000
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

        self.expected_columns = ["id",
                                 "redshift"]
        for r_min, r_max in zip(self.r_mins, self.r_maxes):
            self.expected_columns.append("Mpc%.2ft%.2f_counts" %
                                         (r_min, r_max))
            self.expected_columns.append("Mpc%.2ft%.2f_weights" %
                                         (r_min, r_max))

        self.tmp_file_handle, self.file_name = tempfile.mkstemp(
            dir=os.path.dirname(__file__))

    def tearDown(self):
        del self.tmp_file_handle
        os.remove(self.file_name)

    def test_run(self):
        """Smoke test that the run method runs to completion and outputs
        expected values.
        """
        pm = pair_maker.PairMaker(self.r_mins,
                                  self.r_maxes,
                                  self.z_min,
                                  self.z_max)
        output = pm.run(self.catalog, self.catalog)

        random_index = np.random.int(self.n_objects)

        print(output.iloc[random_index])

        expected_values = np.zeros(len(self.expected_columns))
        for col, val in zip(self.expected_columns, expected_values):
            pd_val = output.iloc[random_idx][col]
            if col == "id":
                self.assertEqual(pd_val, val)
            else:
                self.assertAlmostEqual(pd_val, val)


    def test_output_file(self):
        """Test writing and loading fro the output file. 
        """
        tot_scale_name = "Mpc%.2ft%.2f" % (self.r_min, self.r_max)
        pm = pair_maker.PairMaker(self.r_mins,
                                  self.r_maxes,
                                  self.z_min,
                                  self.z_max,
                                  output_pair_file_name=self.file_name)
        output = pm.run(self.catalog, self.catalog)

        hdf5_file = h5py.File(self.file_name, 'r')

        for r_min, r_max in zip(self.r_mins, self.r_maxes):
            tot_pair_diff = 0
            tot_dist_diff = 0
            for idx in range(self.n_objects):
                data_row = output.iloc[idx]
                dists =  np.exp(
                    hdf5_file["data/%i/%s_log_dists" %
                              (data_row["id"], tot_scale_name)][...])
                scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
                sub_dists = dists[np.logical_and(dists > r_min,
                                                 dists < r_max)]
                n_pairs = len(sub_dists)
                dist_weight = pm._compute_weight(sub_dists).sum()

                pair_diff = 1 - n_pairs / data_row["%s_counts" % scale_name]
                dist_diff = 1 - dist_weight / data_row["%s_weights" %
                                                       scale_name]
                if n_pairs == 0:
                    self.assertEqual(n_pairs,
                                     data_row["%s_counts" % scale_name])
                else:
                    self.assertLess(np.fabs(pair_diff),
                                    3 / data_row["%s_counts" % scale_name])
                if dist_weight == 0:
                    self.assertEqual(dist_weight,
                                     data_row["%s_weights" % scale_name])
                else:
                    self.assertLess(np.fabs(dist_diff),
                                    1 / data_row["%s_counts" % scale_name] *
                                    data_row["%s_weights" % scale_name])
                if np.isfinite(pair_diff):
                    tot_pair_diff += pair_diff
                if np.isfinite(dist_diff):
                    tot_dist_diff += dist_diff
            self.assertAlmostEqual(tot_pair_diff / self.n_objects, 0, places=3)
            self.assertAlmostEqual(tot_dist_diff / self.n_objects, 0, places=3)

    def test_exact_weights(self):
        """Test that the correct pair summary values are computed.
        """
        ids = np.arange(5)
        decs = np.zeros(5)
        ras = np.linspace(0, 500, 5) / 3600
        redshifts = np.full(5, 2.0)
        catalog = {"id": ids,
                   "ra": ras,
                   "dec": decs,
                   "redshift": redshifts}
        pm = pair_maker.PairMaker(self.r_mins,
                                  self.r_maxes,
                                  self.z_min,
                                  self.z_max)
        output = pm.run(catalog, catalog)

        rs = Planck15.comoving_distance(2.0).value * np.radians(ras)
        weights = pm._compute_weight(rs)
        for r_min, r_max in zip(self.r_mins, self.r_maxes):
            scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)

            self.assertEqual(output.iloc[0]["id"], ids[0])
            self.assertEqual(output.iloc[0]["redshift"], redshifts[0])

            tmp_weights = weights[np.logical_and(rs > r_min,
                                                 rs < r_max)]
            self.assertEqual(output.iloc[0]["%s_counts" % scale_name],
                             len(tmp_weights))
            self.assertAlmostEqual(output.iloc[0]["%s_weights" % scale_name],
                                   tmp_weights.sum())


    def test_query_tree(self):
        """Test that the correct number of points are matched in the kdtree.
        """
        pm = pair_maker.PairMaker([1], [10], self.z_min, self.z_max)
        decs = np.zeros(5)
        ras = np.linspace(0, 500, 5) / 3600

        vects = pm._convert_radec_to_xyz(np.radians(ras),
                                         np.radians(decs))
        theta_max = np.radians(450 / 3600)
        dist = 10 / theta_max

        from scipy.spatial import cKDTree
        tree = cKDTree(vects)

        indexes = pm._query_tree(vects[0], tree, dist)
        self.assertEqual(len(indexes), 4)
        self.assertEqual([0, 1, 2, 3], indexes)


if __name__ == "__main__":

    unittest.main()
