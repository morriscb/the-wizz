from astropy.cosmology import Planck15
import h5py
import numpy as np
import os
import pandas as pd
import tempfile
import unittest

from the_wizz import pair_maker
from the_wizz import pdf_maker


class TestPDFMaker(unittest.TestCase):

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
        catalog = {"id": ids,
                   "ra": ras,
                   "dec": decs,
                   "redshift": redshifts}

        self.z_min = 0.01
        self.z_max = 1.1

        pm = pair_maker.PairMaker([1], [10], self.z_min, self.z_max)
        self.pair_counts = pm.run(catalog, catalog)

        self.pairs = pd.DataFrame([
            {"redshift": 0.2, "tot_sample": 10,
             "Mpc1.00t10.00_counts": 5, "Mpc1.00t10.00_weights": 2.5},
            {"redshift": 0.4, "tot_sample": 10,
             "Mpc1.00t10.00_counts": 5, "Mpc1.00t10.00_weights": 2.5},
            {"redshift": 0.6, "tot_sample": 10,
             "Mpc1.00t10.00_counts": 5, "Mpc1.00t10.00_weights": 2.5},
            {"redshift": 0.8, "tot_sample": 10,
             "Mpc1.00t10.00_counts": 5, "Mpc1.00t10.00_weights": 2.5},
            {"redshift": 1.0, "tot_sample": 10,
             "Mpc1.00t10.00_counts": 5, "Mpc1.00t10.00_weights": 2.5},
            {"redshift": 1.2, "tot_sample": 10,
             "Mpc1.00t10.00_counts": 5, "Mpc1.00t10.00_weights": 2.5}])
        self.ref_weights = np.array([1., 0.5, 1., 0.5, 1, 0.5])

    def tearDown(self):
        """
        """
        pass

    def test_run(self):
        """
        """
        pass

    def test_run_bias_mitigation(self):
        """
        """
        pass

    def test_bin_data(self):
        """Test that binning the data and weighting a reference weight works.
        """
        pdf = pdf_maker.PDFMaker(0.0, 1.0, 2)
        binned_data = pdf.bin_data(self.pairs)
        test_data = pd.DataFrame([
            {"mean_redshift": 0.3, "z_min": 0.0, "z_max": 0.5, "dz": 0.5,
             "counts": 10, "weights": 5., "n_ref": 2, "tot_sample": 10},
             {"mean_redshift": 0.7, "z_min": 0.5, "z_max": 1.0, "dz": 0.5,
              "counts": 10, "weights": 5., "n_ref": 2, "tot_sample": 10}])
        for (pd_idx, row), (test_idx, test_row) in zip(binned_data.iterrows(),
                                                       test_data.iterrows()):
            for val, test_val in zip(row, test_row):
                self.assertAlmostEqual(val, test_val)

        binned_data = pdf.bin_data(self.pairs, self.ref_weights)
        test_data = pd.DataFrame([
            {"mean_redshift": (0.2 * 1 + 0.4 * 0.5) / 1.5,
            "z_min": 0.0, "z_max": 0.5, "dz": 0.5,
             "counts": 10, "weights": 3.75, "n_ref": 2, "tot_sample": 10},
            {"mean_redshift": (0.6 * 1 + 0.8 * 0.5) / 1.5,
            "z_min": 0.5, "z_max": 1.0, "dz": 0.5,
             "counts": 10, "weights": 3.75, "n_ref": 2, "tot_sample": 10}])
        for (pd_idx, row), (test_idx, test_row) in zip(binned_data.iterrows(),
                                                       test_data.iterrows()):
            for val, test_val in zip(row, test_row):
                self.assertAlmostEqual(val, test_val)

    def test_compute_correlation(self):
        """Test computing correlations.
        """
        count = np.random.randint(100)
        weight = np.random.uniform(0, 100)
        data = pd.DataFrame([
            {"counts": 2 * count, "weights": 4 * weight, "tot_sample": 1000},
            {"counts": 2 * count, "weights": 4 * weight, "tot_sample": 1000},
            {"counts": 2 * count, "weights": 4 * weight, "tot_sample": 1000},
            {"counts": 2 * count, "weights": 4 * weight, "tot_sample": 1000},
            {"counts": 2 * count, "weights": 4 * weight, "tot_sample": 1000}])
        randoms = pd.DataFrame([
            {"counts": count, "weights": weight, "tot_sample": 2000},
            {"counts": count, "weights": weight, "tot_sample": 2000},
            {"counts": count, "weights": weight, "tot_sample": 2000},
            {"counts": count, "weights": weight, "tot_sample": 2000},
            {"counts": count, "weights": weight, "tot_sample": 2000}])

        pdf = pdf_maker.PDFMaker(self.z_min, self.z_max, 10)
        count_corr, weight_corr = pdf.compute_correlation(data, randoms)

        for idx in range(5):
            count = count_corr[idx]
            weight = weight_corr[idx]
            self.assertEqual(count, 4 - 1)
            self.assertEqual(weight, 8 - 1)

    def test_create_bin_edges(self):
        """Test that all binning types produce predictable results.
        """
        pdf_linear = pdf_maker.PDFMaker(self.z_min, self.z_max, 10, "linear")
        test_linear = np.linspace(self.z_min, self.z_max, 11)
        self.assertEqual(pdf_linear.z_min, self.z_min)
        self.assertEqual(pdf_linear.z_max, self.z_max)
        self.assertEqual(pdf_linear.bins, 10)
        self.assertEqual(pdf_linear.binning_type, "linear")
        for pdf_edge, test_edge in zip(pdf_linear.bin_edges, test_linear):
            self.assertAlmostEqual(pdf_edge, test_edge)

        log_z_min = np.log(1 + self.z_min)
        log_z_max = np.log(1 + self.z_max)
        pdf_log = pdf_maker.PDFMaker(self.z_min, self.z_max, 10, "log")
        test_log = np.linspace(log_z_min, log_z_max, 11)
        self.assertEqual(pdf_log.z_min, self.z_min)
        self.assertEqual(pdf_log.z_max, self.z_max)
        self.assertEqual(pdf_log.bins, 10)
        self.assertEqual(pdf_log.binning_type, "log")
        for pdf_edge, test_edge in zip(np.log(1 + pdf_log.bin_edges),
                                       test_log):
            self.assertAlmostEqual(pdf_edge, test_edge)

        cov_z_min = Planck15.comoving_distance(self.z_min).value
        cov_z_max = Planck15.comoving_distance(self.z_max).value
        pdf_cov = pdf_maker.PDFMaker(self.z_min, self.z_max, 10, "comoving")
        test_cov = np.linspace(cov_z_min, cov_z_max, 11)
        self.assertEqual(pdf_cov.z_min, self.z_min)
        self.assertEqual(pdf_cov.z_max, self.z_max)
        self.assertEqual(pdf_cov.bins, 10)
        self.assertEqual(pdf_cov.binning_type, "comoving")
        for pdf_edge, test_edge in zip(Planck15.comoving_distance(
                                            pdf_cov.bin_edges).value,
                                       test_cov):
            self.assertAlmostEqual(pdf_edge / test_edge - 1, 0, places=6)

        pdf_custom = pdf_maker.PDFMaker(self.z_min,
                                      self.z_max,
                                      np.linspace(1.1, 2.2, 11),
                                      "linear")
        self.assertEqual(pdf_custom.z_min, 1.1)
        self.assertEqual(pdf_custom.z_max, 2.2)
        self.assertEqual(pdf_custom.bins, 10)
        self.assertEqual(pdf_custom.binning_type, "custom")

if __name__ == "__main__":

    unittest.main()
