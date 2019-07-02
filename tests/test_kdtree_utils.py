import numpy as np
from itertools import product
import unittest

from the_wizz.kdtree_utils import SphericalKDTree


class SphericalKDTreeTestCase(unittest.TestCase):

    def setUp(self):
        self.query_point = (40.0, 90.0)  # nothern hemisphere pole
        self.query_limits = (5.0, 10.0)  # angular query in degrees
        self.accuracy = 1e-12  # absolute accuracy requirement
        # generate a grid of data points around the query point
        grid_RAs = np.arange(0.0, 360.0, 18.0)
        grid_DECs = np.arange(70.0, 90.0, 1.0)
        self.grid_points = np.asarray(tuple(product(grid_RAs, grid_DECs)))

    def test__position_sky2sphere(self):
        """
        Test the coordinate conversion of the most trivial cases.
        """
        # test the most trivial points
        self.assertTrue(np.allclose(
            SphericalKDTree._position_sky2sphere(0.0, 0.0),
            (1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(
            SphericalKDTree._position_sky2sphere(90.0, 0.0),
            (0.0, 1.0, 0.0)))
        self.assertTrue(np.allclose(
            SphericalKDTree._position_sky2sphere(0.0, 90.0),
            (0.0, 0.0, 1.0)))
        # test their antipodes
        self.assertTrue(np.allclose(
            SphericalKDTree._position_sky2sphere(180.0, 0.0),
            (-1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(
            SphericalKDTree._position_sky2sphere(270.0, 0.0),
            (0.0, -1.0, 0.0)))
        self.assertTrue(np.allclose(
            SphericalKDTree._position_sky2sphere(0.0, -90.0),
            (0.0, 0.0, -1.0)))
        # test periodicity
        self.assertTrue(np.allclose(
            SphericalKDTree._position_sky2sphere(450.0, 0.0),
            (0.0, 1.0, 0.0)))

    def test__distance_sky2sphere(self):
        """
        Convert a number of angular separations to Euclidean distances between
        points on the unit sphere. Check for symmetry at > 180 deg and
        periodicity at > 360 deg.
        """
        test_dist = (0.0, 60.0, 90.0, 180.0, 270.0, 360.0, 420.0)
        test_truths = (0.0, 1.0, np.sqrt(2.0), 2.0, np.sqrt(2.0), 0.0, 1.0)
        # convert and compare all cases
        for dist, truth in zip(test_dist, test_truths):
            self.assertAlmostEqual(
                SphericalKDTree._distance_sky2sphere(dist), truth)

    def test__distance_sphere2sky(self):
        """
        Convert a number of Euclidean distances between points on the unit
        sphere back to angular separations.
        """
        test_dist = (0.0, 1.0, np.sqrt(2.0), 2.0)
        test_truths = (0.0, 60.0, 90.0, 180.0)
        for dist, truth in zip(test_dist, test_truths):
            self.assertAlmostEqual(
                SphericalKDTree._distance_sphere2sky(dist), truth)

    def test_query_radius(self):
        """
        Query the KDTree derivative around the northern pole within 10 degrees.
        The test grid contains 200 points in this interval.
        """
        rmin, rmax = self.query_limits
        # build and query tree
        self.tree = SphericalKDTree(
            self.grid_points[:, 0], self.grid_points[:, 1])
        idx, dist = self.tree.query_radius(*self.query_point, rmax)
        # check the number of results and distances
        self.assertTrue(np.all(dist <= (rmax + self.accuracy)))
        self.assertEqual(len(idx), 200)

    def test_query_shell(self):
        """
        Query the KDTree derivative around the northern pole within 5 and 10
        degrees. The test grid contains 100 points in this interval.
        """
        rmin, rmax = self.query_limits
        # build and query tree
        self.tree = SphericalKDTree(
            self.grid_points[:, 0], self.grid_points[:, 1])
        idx, dist = self.tree.query_shell(*self.query_point, rmin, rmax)
        # check the number of results and distances
        self.assertTrue(np.all(dist >= (rmin - self.accuracy)))
        self.assertTrue(np.all(dist <= (rmax + self.accuracy)))
        self.assertEqual(len(idx), 100)


if __name__ == "__main__":

    unittest.main()
