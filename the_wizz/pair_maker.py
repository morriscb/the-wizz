
from astropy.cosmology import Planck15
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import InterpolatedUnivariateSpline as InterpSpline


class PairMaker(object):
    """
    """

    def __init__(self,
                 r_mins,
                 r_maxes,
                 z_min,
                 z_max,
                 weight_power=-0.8,
                 distance_metric=None):
        self.r_mins = r_mins
        self.r_maxes = r_maxes
        self.r_min = np.min(r_mins)
        self.r_max = np.max(r_maxes)
        self.z_min = z_min
        self.z_max = z_max

        if distance_metric is None:
            distance_metric = Planck15.comoving_distance
        self.distance_metric = distance_metric

        self.weight_power = weight_power

        self._compute_splines(self.z_min, self.z_max)

    def _compute_splines(self, z_min, z_max):
        """
        """
        redshifts = np.logspace(np.log10(z_min), np.log10(z_max), 100)
        self._z_to_dist = InterpSpline(
            redshifts,
            self.distance_metric(redshifts).value)

        angles = np.logspace(np.log10(np.pi / 4),
                             np.log10(np.radians(0.016 / 3600)),
                             100)
        self._cos_to_ang = InterpSpline(np.cos(angles), angles)

    def run(self, unknown_catalog, reference_catalog, use_unkn_weights=False):
        """
        """
        unkn_vects = self._convert_radec_to_xyz(unknown_catalog["ra"],
                                                unknown_catalog["dec"])
        unkn_tree = cKDTree(unkn_vects)
        unkn_ids = unknown_catalog["id"]
        if use_unkn_weights:
            unkn_weights = unknown_catalog["weight"]
        else:
            unkn_weights = np.ones(len(unkn_ids))

        ref_ids = reference_catalog["id"]
        ref_vects = self._convert_radec_to_xyz(reference_catalog["ra"],
                                               reference_catalog["dec"])
        redshifts = reference_catalog["redshift"]

        output_data = []

        for ref_vect, redshift, ref_id in zip(ref_vects, redshifts, ref_ids):
            if redshift < self.z_min or redshift > self.z_max:
                continue
            dist = self._z_to_dist(redshift)
            theta_max = self.r_max / dist
            unkn_idxs = unkn_tree.query_ball_point(
                ref_vect, np.sqrt(2 - 2 * np.cos(theta_max)))

            tmp_unkn_ids = unkn_ids[unkn_idxs]
            tmp_unkn_vects = unkn_vects[unkn_idxs]
            tmp_unkn_weights = unkn_weights[unkn_idxs]
            cos_thetas = np.dot(tmp_unkn_vects, ref_vect)

            tmp_unkn_dists = self._cos_to_ang(cos_thetas) * dist

            tmp_unkn_sort_args = tmp_unkn_dists.argsort()
            tmp_unkn_ids = tmp_unkn_ids[tmp_unkn_sort_args]
            tmp_unkn_dists = tmp_unkn_dists[tmp_unkn_sort_args]
            tmp_unkn_weights = tmp_unkn_weights[tmp_unkn_sort_args]

            output_row = dict([("id", ref_id), ("redshift", redshift)])

            for r_min, r_max in zip(self.r_mins, self.r_maxes):
                idx_min = np.searchsorted(tmp_unkn_dists, r_min, side="left")
                idx_max = np.searchsorted(tmp_unkn_dists, r_max, side="right")

                bin_unkn_ids = tmp_unkn_ids[idx_min:idx_max]
                bin_unkn_dists = tmp_unkn_dists[idx_min:idx_max]
                bin_unkn_dist_weights = self._compute_weight(bin_unkn_dists)
                bin_unkn_weights = tmp_unkn_weights[idx_min:idx_max]

                output_row["Mpc%.2ft%.2f_counts" % (r_min, r_max)] = \
                    len(bin_unkn_ids)
                output_row["Mpc%.2ft%.2f_weights" % (r_min, r_max)] = \
                    (bin_unkn_dist_weights * bin_unkn_weights).sum()

            output_data.append(output_row)

        return pd.DataFrame(output_data)

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

    def _compute_weight(self, dists):
        """Convert raw distances into a signal matched weight for the
        correlation.

        All weights with distances below 0.001 Mpc will be set to a value of
        0.001 ** ``weight_power``.

        Parameters
        ----------
        dists : `numpy.ndarray`, (N,)
            Distances in Mpc.

        Returns
        -------
        weights : `numpy.ndarray`, (N,)
            Output weights.
        """
        return np.where(dists < 0.001,
                        dists ** self.weight_power,
                        0.001 ** self.weight_power)
