
"""Utility functions for computing single galaxy clustering redshfits using
a k-dimensional tree in galaxy parameter space.
"""

from __future__ import division, print_function, absolute_import

from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree

from the_wizz.pdf_maker_utils import _collapse_multiplex

# TODO:
#     Add option to pickle and loaded pickled CatalogKDtree objects.


def collapse_ids_to_single_estimate(hdf5_pairs_group, pair_data, pdf_maker_obj,
                                    unknown_data, args):
    """This is the heart of the-wizz. It enables the matching of a set of
    catalog ids to the ids stored as pairs to the spectroscopic
    objects. The result of this calculation is a intermediary data product
    containing the density of unknown objects around each reference object
    stored in the PDFMaker data structure class. This specific version is
    for when all the spectra have been pre-loaded in anticipation of running
    a large number of sub-samples as is the case with kdtree recovery.
    ----------------------------------------------------------------------------
    Args:
        hdf5_pairs_group: hdf5 group object containing the pair ids for a fixed
            annulus.
        unknown_data: open fits data containing object ids and relivent weights
        args: ArgumentParser.parse_args object returned from
            input_flags.parse_input_pdf_args
    Returns:
        None
    """
    print("\tpre-loading unknown data...")
    if args.unknown_weight_name is not None:
        unknown_data = unknown_data[
            unknown_data[args.unknown_weight_name] != 0]
    id_array = unknown_data[args.unknown_index_name]
    id_args_array = id_array.argsort()
    id_array = id_array[id_args_array]
    rand_ratio = (
        unknown_data.shape[0]/(1.*hdf5_pairs_group.attrs['n_random_points']))
    if args.unknown_stomp_region_name is not None:
        id_array = [id_array[
            unknown_data[args.unknown_stomp_region_name][id_args_array] ==
            reg_idx]
            for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])]
        tmp_n_region = np.array(
            [id_array[reg_idx].shape[0]
             for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])],
            dtype=np.int_)
        rand_ratio = ((tmp_n_region /
                       (1.*hdf5_pairs_group.attrs['n_random_points'])) *
                      (hdf5_pairs_group.attrs['area'] /
                       hdf5_pairs_group.attrs['region_area']))
    ave_weight = 1.0
    weight_array = np.ones(unknown_data.shape[0], dtype=np.float32)
    if args.unknown_weight_name is not None:
        weight_array = unknown_data[args.unknown_weight_name][id_args_array]
        ave_weight = np.mean(weight_array)
    if args.unknown_stomp_region_name is not None:
        weight_array = [weight_array[
            unknown_data[args.unknown_stomp_region_name][id_args_array] ==
            reg_idx]
            for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])]
        ave_weight = np.array(
            [weight_array[reg_idx].mean()
             for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])],
            dtype=np.float_)
    n_reference = len(hdf5_pairs_group)
    reference_unknown_array = np.empty(n_reference, dtype=np.float32)
    pool = Pool(args.n_processes)
    if args.unknown_stomp_region_name is not None:
        pool_iter = pool.imap(
            _collapse_multiplex,
            [(data_set,
              id_array[pdf_maker_obj.reference_region_array[pair_idx]],
              weight_array[pdf_maker_obj.reference_region_array[pair_idx]],
              args.use_inverse_weighting)
             for pair_idx, data_set in enumerate(pair_data)],
            chunksize=np.int(np.where(args.n_processes > 1,
                             np.log(len(pair_data)), 1)))
    else:
        pool_iter = pool.imap(
            _collapse_multiplex,
            [(data_set, id_array, weight_array, args.use_inverse_weighting)
             for pair_idx, data_set in enumerate(pair_data)],
            chunksize=np.int(np.where(args.n_processes > 1,
                                      np.log(len(pair_data)), 1)))
    print("\t\tcomputing/storing pair count...")
    for pair_idx, reference_value in enumerate(pool_iter):
        reference_unknown_array[pair_idx] = reference_value
    pool.close()
    pool.join()
    pdf_maker_obj.set_reference_unknown_array(reference_unknown_array)
    pdf_maker_obj.scale_random_points(rand_ratio, ave_weight)
    return None


def create_match_data(input_catalog, mag_name_list, other_name_list,
                      use_as_colors):
    kdtree_data_array = np.empty((input_catalog.shape[0], len(mag_name_list)))
    for mag_idx, mag_name in enumerate(mag_name_list):
        kdtree_data_array[:, mag_idx] = input_catalog[mag_name]
        if mag_idx > 0 and use_as_colors:
            kdtree_data_array[:, mag_idx - 1] -= kdtree_data_array[:, mag_idx]
    if use_as_colors:
        kdtree_data_array = np.delete(kdtree_data_array, -1, 1)
    if len(other_name_list) > 0:
        other_data_array = np.empty((input_catalog.shape[0],
                                     len(other_name_list)))
        for other_idx, other_name in enumerate(other_name_list):
            other_data_array[:, other_idx] = input_catalog[other_name]
        kdtree_data_array = np.concatenate(
            (kdtree_data_array, other_data_array), axis=1)
    return kdtree_data_array


class CatalogKDTree(object):
    """Convience class for creating a dataset suitable for a KDTree search, and
    wrapping the scipy KDTree object.
    """
    def __init__(self, input_array):
        """__init__ method preps the internal data storage and creates the
        KDTree.
        ------------------------------------------------------------------------
        Args:
            input_catalog: astropy.io.fits catalog object containing the
                columns of interest
            column_name_list: list of string names of catalog columns to
               consider for the KDTree
            id_column_name: string name of the column containing the indices
        """
        self._internal_array = input_array
        self._normalize_data()
        self._initialize_tree()

    def __call__(self, input_array, k):
        """Given input properties of an object, return the KDTree, array indices
        of the k nearest neighbors.
        ------------------------------------------------------------------------
        Args:
            input_array: float array of object properties (eg fluxes in survey
                bands)
            k: int number of nearest neighbors to return.
        Returns:
            tuple;
                array of integer array indices of objects
                list of quartile and max distances
        """
        tmp_array = (input_array - self._mean_array)/self._std_array
        d, i = self._kd_tree.query(tmp_array, k)
        return i, d[[int(k/4.), int(k/2.), int(3.*k/4.), -1]]

    def k_nearest_ball_point(self, input_array, max_dist):
        """Method to return the KDTree indicies from all points within a fixed
        distance of the point requested. The distance is expressed in sigma of
        the stored data array, i.e. a value of 1 returns all points within 1
        sigma.
        ------------------------------------------------------------------------
        Args:
            input_array: float array of object properties (eg fluxes in survey
                bands)
            max_dist: Maximum radial distance to search from the input point.
        Returns:
            int array of array indices
        """
        tmp_array = (input_array - self._mean_array) / self._std_array
        return self._kd_tree.query_ball_point(tmp_array, max_dist)

    def _initialize_tree(self):
        """Internal method for intilizing the KDTree object.
        -----------------------------------------------------------------------
        Args:
            self
        Returns:
            None
        """
        self._kd_tree = cKDTree(self._internal_array)
        return None

    def _normalize_data(self):
        """Internal method for scaling the data columns stored to a standard
        normal distribution of mean zero and standard deviation of 1.
        ------------------------------------------------------------------------
        Args:
            self
        Returns:
            None
        """
        self._mean_array = self._internal_array.mean(axis=0)
        self._std_array = self._internal_array.std(axis=0)
        for col_idx in xrange(self._internal_array.shape[1]):
            self._internal_array[:, col_idx] = (
                (self._internal_array[:, col_idx] -
                 self._mean_array[col_idx]) / self._std_array[col_idx])
        return None

    def get_mean_array(self):
        return self._mean_array

    def get_std_array(self):
        return self._std_array
