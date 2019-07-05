
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

    def test_output_file(self):
        """
        """
        pass

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
        pm = pair_maker.PairMaker([0.1, 1], [1, 10], self.z_min, self.z_max)
        output = pm.run(catalog, catalog)

        rs = Planck15.comoving_distance(2.0).value * np.radians(ras)
        weights = pm._compute_weight(rs)
        for r_min, r_max in zip([0.1, 1], [1, 10]):
            scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)

            self.assertEqual(output[0]["id"], ids[0])
            self.assertEqual(output[0]["redshift"], redshifts[0])

            tmp_weights = weights(np.logical_and(rs > r_min,
                                                 rs < r_max))
            self.assertEqual(output[0]["%s_counts" % scale_name],
                             len(tmp_weights))
            self.assertAlmostEqual(output[0]["%s_weights" % scale_name],
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
