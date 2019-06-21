
from astropy.cosmology import Planck15
import numpy as np
import h5py
import subprocess
import unittest

from the_wizz import pair_maker


class TestPairMakerUtils(unittest.TestCase):

    def setUp(self):

        np.random.seed(1234)

        self.n_objects = 10000
        decs = np.degrees(np.pi / 2 -
                          np.arccos(np.random.uniform(np.cos(np.radians(1)),
                                                      1,
                                                      size=self.n_objects)))
        ras = np.random.uniform(0, 360, size=self.n_objects)
        redshifts = np.random.lognormal(mean=0.5,
                                        sigma=0.25,
                                        size=self.n_objects)
        ids = np.arange(np.n_objects)
        weights = np.random.uniform(0, 1, size=self.n_objects)

        self.catalog = {"id": ids,
                        "ra": ras,
                        "dec": decs,
                        "redshift": redshifts,
                        "weight": weights}

        self.z_min = 0.01
        self.z_max = 3.0

    def test_splines(self):
        """Test internal splining compared to true expect values.
        """
        pm = pair_maker.PairMaker([1], [10], self.z_min, self.z_max)

        for idx in range(1000):
            angle = np.random.uniform(0, np.pi / 2)
            cos_ang = np.cos(angle)
            self.assertAlmostEqual(
                angle,
                pm._cos_to_theta(cos_ang))

            z = np.random.uniform(self.z_min, self.z_max)
            comov = Planck15.comoving_distnace(z).value
            self.assertAlmostEqual(
                comov,
                pm._z_to_dist(z))


if __name__ == "__main__":

    unittest.main()
