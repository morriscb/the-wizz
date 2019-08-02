
from astropy.cosmology import Planck15
import h5py
import os
import numpy as np
import pandas as pd
import subprocess
import tempfile
import unittest

from the_wizz import pair_maker


class TestPairMaker(unittest.TestCase):

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
        self.output_path = tempfile.mkdtemp(
            dir=os.path.dirname(__file__))

    def tearDown(self):
        job = subprocess.Popen("rm -rf " + self.output_path,
                               shell=True)
        job.wait()
        del job

    def test_run(self):
        """Smoke test that the run method runs to completion and outputs
        expected values.
        """
        pm = pair_maker.PairMaker(self.r_mins,
                                  self.r_maxes,
                                  self.z_min,
                                  self.z_max)
        output = pm.run(self.catalog, self.catalog)

        random_idx = np.random.randint(self.n_objects)

        expected_values = [708,
                           0.6202522969616155,
                           4,
                           6.52884524482144,
                           531,
                           133.259605]
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
                                  n_write_proc=2,
                                  output_pair_file_name=self.output_path,
                                  n_z_bins=4)
        output = pm.run(self.catalog, self.catalog)
        output.set_index("id", inplace=True)

        raw_pair_df = pd.read_parquet("%s/region=0/z_bin=1" % self.output_path)
        raw_pair_df = raw_pair_df.append(pd.read_parquet(
            "%s/region=0/z_bin=2" % self.output_path))
        raw_pair_df = raw_pair_df.append(pd.read_parquet(
            "%s/region=0/z_bin=3" % self.output_path))
        raw_pair_df = raw_pair_df.append(pd.read_parquet(
            "%s/region=0/z_bin=4" % self.output_path))
        raw_pair_df.set_index("ref_id", inplace=True)

        for r_min, r_max in zip(self.r_mins, self.r_maxes):
            tot_pair_diff = 0
            tot_dist_diff = 0
            for ref_id, data_row in output.iterrows():
                raw_data = raw_pair_df.loc[ref_id]
                dists = pair_maker.decompres_distances(
                    raw_data["%s_comp_log_dist" % (tot_scale_name)])
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
                                    2 / data_row["%s_counts" % scale_name])
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
