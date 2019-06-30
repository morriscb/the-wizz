
from astropy.cosmology import Planck15
import h5py
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import InterpolatedUnivariateSpline as InterpSpline

from kdtree_utils import SphericalKDTree


def write_to_pairs_hdf5(data):
    """
    """
    with h5py.File(data["file_name"], "a") as hdf5_file:
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
                    chunks=True, compression='gzip', shuffle=True,
                    scaleoffset=0, compression_opts=9)
                ref_group.create_dataset(
                    dist_name, data=np.log(dist_weights[id_sort_args]),
                    shape=ids.shape, dtype=np.float32,
                    chunks=True, compression='gzip', shuffle=True,
                    scaleoffset=2, compression_opts=9)
    del ref_group
    return None


def callback_error(exception):
    raise exception


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
        unkn_tree = cKDTree(unkn_vects)
        unkn_vects = unkn_tree.tree.data
        unkn_tree = SphericalKDTree(
            unknown_catalog["ra"], unknown_catalog["dec"])
        unkn_ids = unknown_catalog["id"]

        redshifts = reference_catalog["redshift"]
        z_mask = np.logical_and(redshifts > self.z_min, redshifts < self.z_max)
        ref_ids = reference_catalog["id"][z_mask]
        redshifts = reference_catalog["redshift"][z_mask]
        try:
            dists = self.distance_metric(redshifts).value
        except AttributeError:
            dists = self.distance_metric(redshifts)
        except Exception:
            raise ValueError("distance_metric is invalid")

        output_data = []

        self.subproc = None
        if self.output_pair_file_name is not None:
            self.hdf5_writer = Pool(1)

        print("Starting iteration...")
        ref_obj_iter = zip(
            reference_catalog["ra"], reference_catalog["dec"],
            redshifts, dists, ref_ids)
        for ref_ra, ref_dec, redshift, dist, ref_id in ref_obj_iter:
            # Query the unknown tree.
            ang_min = self.r_min / dist  # convert to angular scales
            ang_max = self.r_max / dist
            matched_unkn_ids, matched_unkn_dists = unkn_tree.query_shell(
                ref_ra, ref_dec, ang_min, ang_max)
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
            bin_unkn_dist_weights = unkn_dists[r_mask]

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
            if self.subproc is not None:
                self.subproc.get()
            self.subproc = self.hdf5_writer.apply_async(write_to_pairs_hdf5,
                                                        (hdf5_output_dict,),
                                                        error_callback=callback_error)
            # write_to_pairs_hdf5(hdf5_output_dict)

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
