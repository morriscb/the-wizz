
from astropy.cosmology import Planck15
import h5py
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import InterpolatedUnivariateSpline as InterpSpline


def write_to_pairs_hdf5(data):
    """
    """
    hdf5_file = h5py.File(data["file_name"], "a")
    ref_group = hdf5_file.create_group("data/%i" % data["id"])
    ref_group.attrs["redshift"] = data["redshift"]

    for scale_name in data["scale_names"]:

        id_name = "%s_ids" % scale_name
        dist_name = "%s_dist_weights" % scale_name

        ids = data[id_name]
        dist_weights = data[dist_name].astype(np.float16)

        id_sort_args = ids.argsort()

        if len(ids) <= 0:
            ref_group.create_dataset(
                id_name, shape=ids.shape, dtype=np.uint64)
            ref_group.create_dataset(
                dist_name, shape=ids.shape, dtype=np.float16)
        else:
            ref_group.create_dataset(
                id_name, data=ids[id_sort_args],
                shape=ids.shape, dtype=np.uint64,
                chunks=True, compression='lzf', shuffle=True)
            ref_group.create_dataset(
                dist_name, data=dist_weights[id_sort_args],
                shape=ids.shape, dtype=np.float16,
                chunks=True, compression='lzf', shuffle=True)
    hdf5_file.close()


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
                 distance_metric=None,
                 output_pair_file_name=None):
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

        self.output_pair_file_name = output_pair_file_name

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
        dists = self.distance_metric(redshifts).value

        output_data = []

        if self.output_pair_file_name is not None:
            self.hdf5_writer = Pool(1)

        print("Starting iteration...")
        for ref_vect, redshift, dist, ref_id in zip(ref_vects,
                                                    redshifts,
                                                    dists,
                                                    ref_ids):
            # Query the unknown tree.
            unkn_idxs = np.array(self._query_tree(ref_vect, unkn_tree, dist))

            # Compute angles and convert them to cosmo distances.
            matched_unkn_vects = unkn_vects[unkn_idxs]
            cos_thetas = np.dot(matched_unkn_vects, ref_vect)
            matched_unkn_dists = np.arccos(cos_thetas) * dist
            matched_unkn_ids = unkn_ids[unkn_idxs]

            # Bin data and return counts/sum of weights in bins.
            output_row = self._compute_bin_values(ref_id,
                                                  redshift,
                                                  matched_unkn_ids,
                                                  matched_unkn_dists)
            output_data.append(output_row)

        if self.output_pair_file_name is not None:
            self.hdf5_writer.close()
            self.hdf5_writer.join()

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

        if self.output_pair_file_name is not None:
            hdf5_output_dict = dict([("id", ref_id),
                                     ("redshift", redshift),
                                     ("file_name", self.output_pair_file_name),
                                     ("scale_names", [])])

        for r_min, r_max in zip(self.r_mins, self.r_maxes):
            scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
            r_mask = np.logical_and(unkn_dists > r_min, unkn_dists < r_max)

            bin_unkn_ids = unkn_ids[r_mask]
            bin_unkn_dist_weights = self._compute_weight(unkn_dists[r_mask])

            if self.output_pair_file_name is not None:
                hdf5_output_dict["scale_names"].append(scale_name)
                hdf5_output_dict["%s_ids" % scale_name] = bin_unkn_ids
                hdf5_output_dict["%s_dist_weights" % scale_name] = \
                    bin_unkn_dist_weights

            output_row["%s_counts" % scale_name] = \
                len(bin_unkn_ids)
            output_row["%s_weights" % scale_name] = \
                bin_unkn_dist_weights.sum()

        if self.output_pair_file_name is not None:
            self.hdf5_writer.apply_async(write_to_pairs_hdf5,
                                         (hdf5_output_dict,))

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
        return np.where(dists > 0.001,
                        dists ** self.weight_power,
                        0.001 ** self.weight_power)
