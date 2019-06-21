
from astropy.cosmology import Planck15
import numpy as np
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
        ids = np.arange(self.n_objects)
        weights = np.random.uniform(0, 1, size=self.n_objects)

        self.catalog = {"id": ids,
                        "ra": ras,
                        "dec": decs,
                        "redshift": redshifts,
                        "weight": weights}

        self.z_min = 0.01
        self.z_max = 3.0

    def test_pair_finder(self):
        """
        """
        pass

    def test_splines(self):
        """Test internal splining compared to true expect values.

        We test that the relative output value of the spline is within
        less than a 0.1 percent.
        """
        pm = pair_maker.PairMaker([1], [10], self.z_min, self.z_max)
        angles = np.exp(np.random.uniform(np.log(np.radians(0.1 / 3600)),
                                          np.log(np.pi / 4),
                                          size=10000))
        redshifts = np.random.uniform(self.z_min, self.z_max, size=10000)

        for ang, z in zip(angles, redshifts):
            test_value = 1 - pm._cos_to_ang(np.cos(ang)) / ang
            self.assertAlmostEqual(0, test_value, 3)

            comov = Planck15.comoving_distance(z).value
            test_value = 1 - pm._z_to_dist(z) / comov
            self.assertAlmostEqual(0, test_value, 3)


if __name__ == "__main__":

    unittest.main()
