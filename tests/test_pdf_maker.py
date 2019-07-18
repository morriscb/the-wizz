from astropy.cosmology import Planck15
import h5py
import numpy as np
import os
import pandas as pd
import tempfile
import unittest

from the_wizz import pair_maker
from the_wizz import pdf_maker


class TestPDFMakerUtils(unittest.TestCase):

    def setup(self):
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

        self.z_min = 0.0
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

    def bin_data(self):
        """
        """
        pass

    def compute_correlation(self):
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
            self.assertEqual(count, 4)
            self.assertEqual(count, 8)

    def test_create_bin_edges(self):
        pass


if __name__ == "__main__":

    unittest.main()
