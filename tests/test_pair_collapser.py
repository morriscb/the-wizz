
import os
import numpy as np
import pandas as pd
import subprocess
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
        median_dec = np.median(decs)
        ras = np.random.uniform(0, 360, size=self.n_objects)
        redshifts = np.random.lognormal(mean=-1,
                                        sigma=0.5,
                                        size=self.n_objects)
        ids = np.arange(self.n_objects, dtype=np.uint64)
        regions = np.where(decs > median_dec, 1, 0)
        self.catalog = {"id": ids,
                        "ra": ras,
                        "dec": decs,
                        "redshift": redshifts,
                        "region": regions}

        self.z_min = 0.05
        self.z_max = 3.0

        self.r_mins = [1,]
        self.r_maxes = [10,]
        self.r_min = np.min(self.r_mins)
        self.r_max = np.max(self.r_maxes)

        self.weight_power = -0.8

        self.expected_columns = ["ref_id",
                                 "redshift"]
        for r_min, r_max in zip(self.r_mins, self.r_maxes):
            self.expected_columns.append("Mpc%.2ft%.2f_count" %
                                         (r_min, r_max))
            self.expected_columns.append("Mpc%.2ft%.2f_weight" %
                                         (r_min, r_max))
        self.output_path = tempfile.mkdtemp(
            dir=os.path.dirname(__file__))

    def tearDown(self):
        job = subprocess.Popen("rm -rf " + self.output_path,
                               shell=True)
        job.wait()
        del job

    def test_run(self):
        """Test the full run method and that it outputs values similar to
        the straight correlation.
        """
        pm_output = self._run_pair_maker()

        self._test_run_different_n_proc(0, pm_output)
        self._test_run_different_n_proc(4, pm_output)

    def _run_pair_maker(self):
        """Run pair maker and create output

        Returns
        -------
        pm_output : `pandas.DataFrame`
            Return the output of PairMaker.
        """
        pm = pair_maker.PairMaker(self.r_mins,
                                  self.r_maxes,
                                  self.z_min,
                                  self.z_max,
                                  self.weight_power,
                                  output_pairs=self.output_path)
        pm_output = []
        for region in [0, 1]:
            mask = self.catalog["region"] == region
            region_cat = {"id": self.catalog["id"][mask],
                          "ra": self.catalog["ra"][mask],
                          "dec": self.catalog["dec"][mask],
                          "redshift": self.catalog["redshift"][mask],
                          "region": self.catalog["region"][mask]}
            pm_output.append(pm.run(region_cat, region_cat))
        pm_output = pd.concat(pm_output)
        pm_output.set_index("ref_id", inplace=True)

        return pm_output

    def _test_run_different_n_proc(self, n_proc, pm_output):
        """Test different n_proc levels.

        Parameters
        ----------
        n_proc : `int`
            Number of sub processes to use.
        pm_output : `pandas.DataFrame`
            Output from `PairMaker.run`
        """
        pc = pair_collapser.PairCollapser(self.output_path,
                                          self.r_mins,
                                          self.r_maxes,
                                          self.weight_power,
                                          n_proc=n_proc)
        pc_output = pc.run(self.catalog)
        pc_output.set_index("ref_id", inplace=True)

        self.assertEqual(len(pm_output), len(pc_output))

        for ref_id, ref_row in pm_output.iterrows():
            pc_row = pc_output.loc[ref_id]
            self.assertAlmostEqual(ref_row["Mpc1.00t10.00_count"],
                                   pc_row["Mpc1.00t10.00_count"])
            self.assertAlmostEqual(ref_row["Mpc1.00t10.00_weight"] / 
                                   pc_row["Mpc1.00t10.00_weight"] - 1,
                                   0,
                                   places=3)

    def test_collapse_pairs(self):
        """Test reading from parquet and computing correlations.
        """
        pm_output = self._run_pair_maker()

        mask = self.catalog["region"] == 0
        data = {"unkn_ids": self.catalog["id"][mask],
                "unkn_weights": np.ones(mask.sum()),
                "tot_sample": mask.sum(),
                "r_mins": self.r_mins,
                "r_maxes": self.r_maxes,
                "file_name": self.output_path,
                "region": "region=%i" % 0,
                "z_bin": self.output_path + "/region=0/z_bin=25",
                "weight_power": self.weight_power,
                "ave_unkn_weight": 1.0}
        pc_output = pair_collapser.collapse_pairs(data)
        pc_output.set_index("ref_id", inplace=True)

        for ref_id, pc_row in pc_output.iterrows():
            ref_row = pm_output.loc[ref_id]
            self.assertAlmostEqual(ref_row["Mpc1.00t10.00_count"],
                                   pc_row["Mpc1.00t10.00_count"])
            self.assertAlmostEqual(ref_row["Mpc1.00t10.00_weight"] / 
                                   pc_row["Mpc1.00t10.00_weight"] - 1,
                                   0,
                                   places=3)

    def test_collapse_pairs_ref_id(self):
        """Test that masking of pairs is working.
        """
        ref_row = {"redshift": 0.5,
                   "region": 0}
        comp_dists = pair_maker.compress_distances(np.linspace(0.1, 10, 20))
        pair_data = pd.DataFrame(
            data={"unkn_id": np.arange(20),
                  "comp_log_dist": comp_dists})
        unkn_ids = np.array([3, 5, 7, 9, 21])
        unkn_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        output = pair_collapser.collapse_pairs_ref_id(ref_row,
                                                      pair_data,
                                                      unkn_ids,
                                                      unkn_weights,
                                                      self.r_mins,
                                                      self.r_maxes,
                                                      self.weight_power)
        matched_dists, matched_weights = pair_collapser.find_pairs(
            pair_data["unkn_id"].to_numpy(),
            unkn_ids,
            pair_maker.distance_weight(
                pair_maker.decompress_distances(
                    pair_data["comp_log_dist"].to_numpy()),
                self.weight_power),
            unkn_weights)
        
        self.assertEqual(output["Mpc1.00t10.00_count"], len(matched_dists))
        self.assertEqual(output["Mpc1.00t10.00_weight"],
                         (matched_dists * matched_weights).sum())

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
