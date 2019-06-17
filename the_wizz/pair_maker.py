
from astropy.cosmology import Planck15
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline


class PairMaker(object):
    """
    """

    def __init__(self, r_mins, r_maxes, z_min, z_max):
        self.r_min = np.min(r_mins)
        self.r_max = np.max(r_maxes)
        self.z_min = z_min
        self.z_max = z_max

        self._compute__splines(np.min(r_mins), np.max(r_maxes))

    def _compute_splines(self, z_min, z_max):
        """
        """
        redshifts = np.logspace(np.log10(z_min), np.log10(z_max), 1000)
        self._z_to_dist = UnivariateSpline(
            redshifts,
            Planck15.comoving_distnace(redshifts).values)
        angles = np.linspace(0, np.pi / 2, 10000)
        self._cos_to_theta = UnivariateSpline(np.cos(angles), angles)

    def run(self, unknown_catalog, reference_catalog):
        """
        """
        unkn_vects = self._convert_radec_to_xyz(unknown_catalog["ra"],
                                                unknown_catalog["dec"])
        unkn_tree = cKDTree(unkn_vects)
        unkn_ids = unknown_catalog["id"]

        ref_ids = reference_catalog["id"]
        ref_vects = self._convert_radec_to_xyz(reference_catalog["ra"],
                                               reference_catalog["dec"])
        dists = self._z_to_dist(reference_catalog["z"])

        for ref_vect, dist, ref_id in zip(ref_vects, dists, ref_ids):
            theta_max = self.r_max / dist
            unkn_idxs = unkn_tree.query_ball_point(ref_vect,
                                                   2 - 2 * np.cos(theta_max))

            tmp_unkn_ids = unkn_ids[unkn_idxs]
            tmp_unkn_vects = unkn_vects[unkn_idxs]
            cos_thetas = np.dot(tmp_unkn_vects, ref_vect)
            tmp_unkn_dists = self._cos_to_theta(cos_thetas) * dist

            tmp_unkn_sort_args = tmp_unkn_dists.argsort()
            tmp_unkn_ids = tmp_unkn_ids[tmp_unkn_sort_args]
            tmp_unkn_dists = tmp_unkn_dists[tmp_unkn_sort_args]

            for r_min, r_max in zip(self.r_mins, self.r_maxes):
                idx_min = np.searchsorted(tmp_unkn_dists, r_min, side="left")
                idx_max = np.searchsorted(tmp_unkn_dists, r_max, side="right")

                bin_unkn_dists = tmp_unkn_dists[]
                
    def _convert_radec_to_xyz(self, ras, decs):
        """Convert RA/DEC positions to points on the unit sphere.

        Parameters
        ----------
        ras : `numpy.ndarray`, (N,)
            Right assertion coordinate in radians
        decs : `numpy.ndarray`, (N,)
            Declination coordinate in radians

        Returns
        -------
        vectors : `numpy.ndarray`, (N, 3)
            Array of points on the unit sphere.
        """
        vectors = np.empty((len(ras), 3))

        vectors[:, 2] = np.sin(decs)

        sintheta = np.cos(decs)
        vectors[:, 0] = np.cos(ras) * sintheta
        vectors[:, 1] = np.sin(ras) * sintheta

        return vectors

    def _make_kdtree(self, ras, decs):
        """Create a spatial search tree given input ra/decs.

        Parameters
        ----------
        ras : `numpy.ndarray`, (N,)
            Right assertion coordinate in radians
        decs : `numpy.ndarray`, (N,)
            Declination coordinate in radians

        Returns
        -------
        spatial_tree : `scipy.spatial.cKDTree`
            searchable spatial tree on the unit-sphere.
        """
        vectors = self._convert_radec_to_xyz(ras, decs)
        return cKDTree(vectors)
