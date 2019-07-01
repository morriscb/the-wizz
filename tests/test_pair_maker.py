
from astropy.cosmology import Planck15
import numpy as np
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

    def test_run(self):
        """Test that the run method runs to completion and outputs expected
        values.
        """
        pm = pair_maker.PairMaker([0.1, 1], [1, 10], self.z_min, self.z_max)
        output = pm.run(self.catalog, self.catalog)

    def test_exact_weights(self):
        """
        """
        pass


if __name__ == "__main__":

    unittest.main()
