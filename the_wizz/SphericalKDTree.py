import numpy as np
from scipy.spatial import minkowski_distance, cKDTree


class SphericalKDTree(object):
    """
    SphericalKDTree(RA, DEC, leaf_size=16)

    A binary search tree based on scipy.spatial.cKDTree that works with
    celestial coordinates. Provides methods to find pairs within angular
    apertures (ball) and annuli (shell). Data is internally represented on a
    unit-sphere in three dimensions (x, y, z).

    Parameters
    ----------
    RA : array_like
        List of right ascensions in degrees.
    DEC : array_like
        List of declinations in degrees.
    leafsize : int
        The number of points at which the algorithm switches over to
        brute-force.
    """

    def __init__(self, RA, DEC, leafsize=16):
        # convert angular coordinates to 3D points on unit sphere
        pos_sphere = self._position_sky2sphere(RA, DEC)
        self.tree = cKDTree(pos_sphere, leafsize)

    @staticmethod
    def _position_sky2sphere(RA, DEC):
        """
        _position_sky2sphere(RA, DEC)

        Maps celestial coordinates onto a unit-sphere in three dimensions
        (x, y, z).

        Parameters
        ----------
        RA : float or array_like
            Single or list of right ascensions in degrees.
        DEC : float or array_like
            Single or list of declinations in degrees.

        Returns
        -------
        pos_sphere : array like
            Data points (x, y, z) representing input points on the unit-sphere,
            shape of output is (3,) for a single input point or (N, 3) for a
            set of N input points.
        """
        ras_rad = np.deg2rad(RA)
        decs_rad = np.deg2rad(DEC)
        try:
            pos_sphere = np.empty((len(RA), 3))
        except TypeError:
            pos_sphere = np.empty((1, 3))
        cos_decs = np.cos(decs_rad)
        pos_sphere[:, 0] = np.cos(ras_rad) * cos_decs
        pos_sphere[:, 1] = np.sin(ras_rad) * cos_decs
        pos_sphere[:, 2] = np.sin(decs_rad)
        return np.squeeze(pos_sphere)

    @staticmethod
    def _distance_sky2sphere(dist_sky):
        """
        _distance_sky2sphere(dist_sky)

        Converts angular separation in celestial coordinates to the
        Euclidean distance in (x, y, z) space.

        Parameters
        ----------
        dist_sky : float or array_like
            Single or list of separations in celestial coordinates.

        Returns
        -------
        dist_sphere : float or array_like
            Celestial separation converted to (x, y, z) Euclidean distance.
        """
        dist_sky_rad = np.deg2rad(dist_sky)
        dist_sphere = np.sqrt(2.0 - 2.0 * np.cos(dist_sky_rad))
        return dist_sphere

    @staticmethod
    def _distance_sphere2sky(dist_sphere):
        """
        _distance_sphere2sky(dist_sphere)

        Converts Euclidean distance in (x, y, z) space to angular separation in
        celestial coordinates.

        Parameters
        ----------
        dist_sphere : float or array_like
            Single or list of Euclidean distances in (x, y, z) space.

        Returns
        -------
        dist_sky : float or array_like
            Euclidean distance converted to celestial angular separation.
        """
        dist_sky_rad = np.arccos(1.0 - dist_sphere**2 / 2.0)
        dist_sky = np.rad2deg(dist_sky_rad)
        return dist_sky

    def query_radius(self, RA, DEC, r):
        """
        query_radius(RA, DEC, r)

        Find all data points within an angular aperture r around a reference
        point with coordiantes (RA, DEC) obeying the spherical geometry.

        Parameters
        ----------
        RA : float
            Right ascension of the reference point in degrees.
        DEC : float
            Declination of the reference point in degrees.
        r : float
            Maximum separation of data points from the reference point.

        Returns
        -------
        idx : array_like
            Positional indices of matching data points in the search tree data
            with sepration < r.
        dist : array_like
            Angular separation of matching data points from reference point.
        """
        point_sphere = self._position_sky2sphere(RA, DEC)
        # find all points that lie within r
        r_sphere = self._distance_sky2sphere(r)
        idx = self.tree.query_ball_point(point_sphere, r_sphere)
        # compute pair separation
        dist_sphere = minkowski_distance(self.tree.data[idx], point_sphere)
        dist = self._distance_sphere2sky(dist_sphere)
        return idx, dist

    def query_shell(self, RA, DEC, rmin, rmax):
        """
        query_radius(RA, DEC, r)

        Find all data points within an angular annulus rmin <= r < rmax around
        a reference point with coordiantes (RA, DEC) obeying the spherical
        geometry.

        Parameters
        ----------
        RA : float
            Right ascension of the reference point in degrees.
        DEC : float
            Declination of the reference point in degrees.
        rmin : float
            Minimum separation of data points from the reference point.
        rmax : float
            Maximum separation of data points from the reference point.

        Returns
        -------
        idx : array_like
            Positional indices of matching data points in the search tree data
            with rmin <= sepration < rmax.
        dist : array_like
            Angular separation of matching data points from reference point.
        """
        # find all points that lie within rmax
        idx, dist = self.query_radius(RA, DEC, rmax)
        # remove pairs with r >= rmin
        dist_mask = dist >= rmin
        idx = np.compress(dist_mask, idx)
        dist = np.compress(dist_mask, dist)
        return idx, dist


if __name__ == "__main__":

    N = 10000

    # test distance conversion
    for dist_sky in np.linspace(0.0, 360.0):
        dist_sphere = SphericalKDTree._distance_sky2sphere(dist_sky)
        back_trans = SphericalKDTree._distance_sphere2sky(dist_sphere)
        dist_sky_wrapped = np.minimum(dist_sky, 360.0 - dist_sky)
        assert(np.isclose(dist_sky_wrapped, back_trans))

    # test data conversion
    rand_ra = np.random.uniform(0.0, 180.0, size=N)
    rand_delta_dec = np.random.uniform(0.0, 180.0, size=N)
    # compute two points that lie on a great circle
    pos_sky1 = (rand_ra, 90.0 - rand_delta_dec / 2.0)
    pos_sky2 = (180.0 + rand_ra, 90.0 - rand_delta_dec / 2.0)
    pos_sphere1 = SphericalKDTree._position_sky2sphere(*pos_sky1)
    pos_sphere2 = SphericalKDTree._position_sky2sphere(*pos_sky2)
    # these points must be separated by dist_sky
    dist_sphere = minkowski_distance(pos_sphere1, pos_sphere2)
    back_trans = SphericalKDTree._distance_sphere2sky(dist_sphere)
    assert(np.isclose(rand_delta_dec, back_trans).all())

    # random sky coordinates
    RAs = np.random.uniform(0.0, 360.0, size=N)
    DECs = np.random.uniform(70.0, 89.99, size=N)
    rmin, rmax = 0.5, 1.0  # degrees
    point_sky = (44.0, 90.0)  # northern pole
    # compute using brute force
    pos_sphere = SphericalKDTree._position_sky2sphere(RAs, DECs)
    point_sphere = SphericalKDTree._position_sky2sphere(*point_sky)
    dist_sphere = minkowski_distance(pos_sphere, point_sphere)
    dist_sky = SphericalKDTree._distance_sphere2sky(dist_sphere)
    mask = (dist_sky >= rmin) & (dist_sky < rmax)
    idx_bruteforce = set(np.arange(N)[mask])
    dist_bruteforce = dist_sky[mask]
    # test SphericalKDTree
    tree = SphericalKDTree(RAs, DECs)
    idx_kd, dist_kd = tree.query_shell(
        point_sky[0], point_sky[1], rmin, rmax)
    idx_kd = set(idx_kd)
    assert(dist_kd.min() >= rmin)
    assert(dist_kd.max() < rmax)
    assert(idx_kd == idx_bruteforce)
