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

    def tearDown(self):
        """
        """
        pass

    def test_run(self):
        """
        """
        pass

    def test_bin_data(self):
        """
        """
        pdf = pdf_maker.PDFMaker(self.z_min, self.z_max, 10)

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
