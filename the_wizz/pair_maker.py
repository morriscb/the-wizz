
from astropy.cosmology import Planck15
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import InterpolatedUnivariateSpline as InterpSpline


class PairMaker(object):
    """Class for computing distance weighted correlations of a reference sample
    with known redshift against a sample with unknown redshifts.

    Parameters
    ----------
    r_mins : `list` of `float`s
        List of bin edge minimums in Mpc.
    r_maxes : `list` of `float`s
        List of bin edge maximums in Mpc.
    z_min : `float`
        Minimum redshift of the reference sample to consider.
    z_max : `float`
        Maximum redshift of the reference sample to consider.
    weight_power : `float`
        Expected power-law slope of the projected correlation function. Used
        for signal matched weighting.
    distance_metric : `astropy.cosmology.LambdaCDM.<distance>`
        Cosmological distance metric to use for all calculations. Should be
        either comoving_distance or angular_diameter_distance. Defaults to
        the Planck15 cosmology and comoving metric.
    """
    def __init__(self,
                 r_mins,
                 r_maxes,
                 z_min=0.01,
                 z_max=5.00,
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
        """Create splines for lookup between redshift and distance as well
        ass cos(theta) to theta.

        Minimum theta is 0.016 arcseconds. Max is 45 degrees.

        Parameters
        ----------
        z_min : `float`
            Minimum redshift to create distance spline
        z_max : `float`
        """
        redshifts = np.logspace(np.log10(z_min), np.log10(z_max), 100)
        self._z_to_dist = InterpSpline(
            redshifts,
            self.distance_metric(redshifts).value)

        angles = np.logspace(np.log10(np.pi / 4),
                             np.log10(np.radians(0.016 / 3600)),
                             100)
        self._cos_to_ang = InterpSpline(np.cos(angles), angles)

    def run(self, reference_catalog, unknown_catalog, use_unkn_weights=False):
        """Find the (un)weighted pair counts between reference and unknown
        catalogs.

        Parameters
        ----------
        reference_catalog : 'dict' of `numpy.ndarray`
            Dictionary containing arrays 
        """
        unkn_vects = self._convert_radec_to_xyz(
            np.radians(unknown_catalog["ra"]),
            np.radians(unknown_catalog["dec"]))
        unkn_tree = cKDTree(unkn_vects)
        unkn_ids = unknown_catalog["id"]

        redshifts = reference_catalog["redshift"]
        z_mask = np.logical_and(redshifts > self.z_min, redshifts < self.z_max)
        ref_ids = reference_catalog["id"][z_mask]
        ref_vects = self._convert_radec_to_xyz(
            np.radians(reference_catalog["ra"][z_mask]),
            np.radians(reference_catalog["dec"][z_mask]))
        redshifts = reference_catalog["redshift"][z_mask]
        dists = self._z_to_dist(redshifts)

        output_data = []

        for ref_vect, redshift, dist, ref_id in zip(ref_vects,
                                                    redshifts,
                                                    dists,
                                                    ref_ids):
            # Query the unknown tree.
            unkn_idxs = np.array(self._query_tree(ref_vect, unkn_tree, dist))

            # Compute angles and convert them to cosmo distances.
            matched_unkn_vects = unkn_vects[unkn_idxs]
            cos_thetas = np.dot(matched_unkn_vects, ref_vect)
            matched_unkn_dists = self._cos_to_ang(cos_thetas) * dist
            matched_unkn_ids = unkn_ids[unkn_idxs]

            # Bin data and return counts/sum of weights in bins.
            output_row = self._compute_bin_values(ref_id,
                                                  redshift,
                                                  matched_unkn_ids,
                                                  matched_unkn_dists)
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

    def _query_tree(self, ref_vect, unkn_tree, dist):
        """Query the kdtree for all points within the maximum r value at a
        given redshift/distance.

        Parameters
        ----------
        ref_vecct : `numpy.ndarray`, (3,)
            Position to center ball tree search on.
        unkn_tree : `scipy.spatial.cKDTree`
            Searchable kdtree containing points to correlate with.
        dist : `float`
            Distance from observer to the reference object at redshift, z.

        Returns
        -------
        output_indexes : `list` of `int`s
            List of integer index lookups into the array the tree was created
            with.
        """
        theta_max = self.r_max / dist
        return unkn_tree.query_ball_point(
            ref_vect,
            np.sqrt(2 - 2 * np.cos(theta_max)))

    def _compute_bin_values(self,
                            ref_id,
                            redshift,
                            unkn_ids,
                            unkn_dists):
        """
        """
        output_row = dict([("id", ref_id), ("redshift", redshift)])

        for r_min, r_max in zip(self.r_mins, self.r_maxes):
            r_mask = np.logical_and(unkn_dists > r_min, unkn_dists < r_max)

            bin_unkn_ids = unkn_ids[r_mask]
            bin_unkn_dist_weights = self._compute_weight(unkn_dists[r_mask])

            output_row["Mpc%.2ft%.2f_counts" % (r_min, r_max)] = \
                len(bin_unkn_ids)
            output_row["Mpc%.2ft%.2f_weights" % (r_min, r_max)] = \
                bin_unkn_dist_weights.sum()

        return output_row

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
