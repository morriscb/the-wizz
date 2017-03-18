
"""Utility functions for collapsing the id arrays stored in a The-wiZZ HDF5 file
into a clustering redshift estimate given an input sample.
"""

from __future__ import division, print_function, absolute_import

import pickle

from multiprocessing import Pool
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iu_spline

from the_wizz import core_utils


def _create_linear_redshift_bin_edges(z_min, z_max, n_bins):
    """Simple utility for computing redshift bins that are linearly spaced in
    redshift. Not recommened for use if your concern is maximum signal to noise.
    ----------------------------------------------------------------------------
    args:
        z_min: float, minimum redshift to bin from
        z_max: float, maximum redshift to bin to
        n_bins: int, number of bins
    returns:
        numpy.array of type float and shape (n_bins,) containing the lower bin
        edges. The n_bin + 1 edge is equal to z_max.
    """
    return np.linspace(z_min, z_max, n_bins + 1)[:-1]

def _create_adaptive_redshift_bin_edges(z_min, z_max, n_bins, redshift_array):
    """Simple utility for computing redshift bins that delivers a consistent
    number of spectroscopic objects per bin.
    ----------------------------------------------------------------------------
    args:
        z_min: float, minimum redshift to bin from
        z_max: float, maximum redshift to bin to
        n_bins: int, number of bins
        redshift_array: numpy.array float of the spectroscopic redshifts.
    returns:
        numpy.array of type float and shape (n_bins,) containing the lower bin
        edges. The n_bin + 1 edge is equal to z_max.
    """
    useable_z_array = redshift_array[np.logical_and(redshift_array >= z_min,
                                                    redshift_array < z_max)]
    useable_z_array.sort()
    return useable_z_array[np.arange(0, useable_z_array.shape[0],
                                     useable_z_array.shape[0]/n_bins,
                                     dtype=np.int_)]

def _create_logspace_redshift_bin_edges(z_min, z_max, n_bins):
    """Simple utility for computing redshift bins that are equally spaced in
    comoving, line of sight distance. This creates bins that have a smoother
    bias versus redshift.
    ----------------------------------------------------------------------------
    args:
        z_min: float, minimum redshift to bin from
        z_max: float, maximum redshift to bin to
        n_bins: int, number of bins
    returns:
        numpy.array of type float and shape (n_bins,) containing the lower bin
        edges. The n_bin + 1 edge is equal to z_max.
    """
    log_min = np.log(1 + z_min)
    log_max = np.log(1 + z_max)
    return np.exp(np.linspace(log_min, log_max, n_bins + 1)[:-1]) - 1.0

def _create_comoving_redshift_bin_edges(z_min, z_max, n_bins):
    """Simple utility for computing redshift bins that are equally spaced in
    comoving, line of sight distance. This creates bins that have a smoother
    bias versus redshift.
    ----------------------------------------------------------------------------
    args:
        z_min: float, minimum redshift to bin from
        z_max: float, maximum redshift to bin to
        n_bins: int, number of bins
    returns:
        numpy.array of type float and shape (n_bins,) containing the lower bin
        edges. The n_bin + 1 edge is equal to z_max.
    """
    comov_min = core_utils.WMAP5.comoving_distance(z_min).value
    comov_max = core_utils.WMAP5.comoving_distance(z_max).value

    comov_dist_to_redsihft_spline = _make_redshift_spline(z_min, z_max)
    
    return comov_dist_to_redsihft_spline(
        np.linspace(comov_min, comov_max, n_bins + 1)[:-1])

def _make_redshift_spline(z_min, z_max):
    """Utility function for creating a spline for comoving distance to
    redshift.
    """
    redshift_array = np.linspace(
        np.min(z_min - 1e-8, 0.0), z_max + 1e-8, 1000)
    comov_array = core_utils.WMAP5.comoving_distance(redshift_array)
    comov_dist_to_redshift_spline = iu_spline(comov_array, redshift_array)
    return comov_dist_to_redshift_spline

def collapse_ids_to_single_estimate(hdf5_pairs_group, pdf_maker_obj,
                                    unknown_data, args):
    """This is the heart of The-wiZZ. It enables the matching of a set of
    catalog ids to the ids stored as pairs to the spectroscopic objects. The
    result of this calculation is a intermediary data product containing the
    density of unknown objects around each target object stored in the PDFMaker
    data structure class.
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
    # First we load the the ids from the input fits data using the columns
    # names the user has provided and scale the randoms to the correct ammount.
    # Object ids are also sorted in increasing id for later binary search.
    if args.unknown_weight_name is not None:
        unknown_data = unknown_data[unknown_data[args.unknown_weight_name] != 0]
    id_array = unknown_data[args.unknown_index_name]
    id_args_array = id_array.argsort()
    id_array = id_array[id_args_array]
    rand_ratio = (unknown_data.shape[0] /
                  hdf5_pairs_group.attrs['n_random_points'])
    # If the user specifies the name of a STOMP region column, break up the data
    # into the individual regions.
    if args.unknown_stomp_region_name is not None:
        id_array = [id_array[
            unknown_data[args.unknown_stomp_region_name][id_args_array] ==
            reg_idx]
            for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])]
        tmp_n_region = np.array(
            [id_array[reg_idx].shape[0]
             for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])],
            dtype=np.int_)
        rand_ratio = (
            (tmp_n_region/hdf5_pairs_group.attrs['n_random_points']) *
            (hdf5_pairs_group.attrs['area'] /
             hdf5_pairs_group.attrs['region_area']))
    # Now that we loaded the ids we can load unknown object weights in the same
    # way.
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
    # Prime our output array.
    n_target = len(hdf5_pairs_group)
    target_unknown_array = np.empty(n_target, dtype=np.float32)
    # Set some interation variables and start the loop over the target objects.
    pair_start = 0
    hold_pair_start = 0
    pair_data = []
    # Initialize the workers.
    pool = Pool(args.n_processes)
    while hold_pair_start < n_target:
        # This is where the heavy lifting of pair matching happens. There are
        # two options for if we use regions or not. If we use regions, the
        # code runs significantly quicker.
        if pair_data:
            print("\t\tmatching pairs: starting targets %i-%i..." %
                  (hold_pair_start, hold_pair_start + args.n_target_load_size))
            # If we are using regions.
            if args.unknown_stomp_region_name is not None:
                pool_iter = pool.imap(
                    _collapse_multiplex,
                    [(data_set,
                      id_array[
                          pdf_maker_obj.target_region_array[hold_pair_start +
                                                            pair_idx]],
                      weight_array[
                          pdf_maker_obj.target_region_array[hold_pair_start +
                                                            pair_idx]],
                      args.use_inverse_weighting)
                     for pair_idx, data_set in enumerate(pair_data)],
                    chunksize=np.int(np.where(
                        args.n_processes > 1, np.log(len(pair_data)), 1)))
            # Not using regions.
            else:
                pool_iter = pool.imap(
                    _collapse_multiplex,
                    [(data_set, id_array, weight_array,
                      args.use_inverse_weighting)
                     for pair_idx, data_set in enumerate(pair_data)],
                    chunksize=np.int(np.where(
                        args.n_processes > 1, np.log(len(pair_data)), 1)))
        # Now that we've submitted the jobs to the workers we load the next
        # batch of data.
        print("\t\tloading next pairs...")
        pair_data = _load_pair_data(hdf5_pairs_group, pair_start,
                                    args.n_target_load_size)
        # This is a cludgy way of testing to see that pool_iter exists and we
        # can then iterate and store our results from the workers.
        try:
            type(pool_iter)
            print("\t\tcomputing/storing pair count...")
            for pair_idx, target_value in enumerate(pool_iter):
                target_unknown_array[hold_pair_start + pair_idx] = target_value
        except UnboundLocalError:
            pass
        # Increase the load and store variables.
        hold_pair_start = pair_start
        pair_start += args.n_target_load_size
    # Close the workers.
    pool.close()
    pool.join()
    # Store the results in our PDFMaker class object.
    pdf_maker_obj.set_target_unknown_array(target_unknown_array)
    pdf_maker_obj.scale_random_points(rand_ratio, ave_weight)
    return None

def _collapse_multiplex(input_tuple):
    """Function for matching indices and calculating the over densities of a
    specific set of unknown objects around the target object. This specific
    function is meant to be used within the context of python multiprocessing.
    ----------------------------------------------------------------------------
    Args:
        input_tuple: tuple of arrays and values specifying the current data to
            consider for the target object.
    Returns:
        float n_points
    """
    (data_set, id_array, weight_array,
     use_inverse_weighting) = input_tuple
    id_data_set, inv_data_set = data_set
    if id_data_set.shape[0] == 0 or id_array.shape[0] == 0:
        return 0.0
    # Since the ids around the target are partially localized spatially a cut
    # in id is also a cut spatially. Here we take advantage of this.
    start_idx = np.searchsorted(id_array, id_data_set[0])
    end_idx = np.searchsorted(id_array, id_data_set[-1], side='right')
    if start_idx == end_idx:
        return 0.0
    if start_idx < 0:
        start_idx = 0
    if end_idx > id_array.shape[0]:
        end_idx = id_array.shape[0]
    tmp_n_points = 0.0
    # We test to see which array is longer, the hdf5 id array or our unknown
    # array. We loop over the shorter of the two. This can yield a significant
    # speed boost for small scales or sparse samples.
    # TODO: change this so most of it happens within numpy.
    if id_array[start_idx:end_idx].shape[0] <= id_data_set.shape[0]:
        for obj_id, weight in zip(id_array[start_idx:end_idx],
                                  weight_array[start_idx:end_idx]):
            sort_idx = np.searchsorted(id_data_set, obj_id)
            if sort_idx >= id_data_set.shape[0] or sort_idx < 0:
                continue
            if id_data_set[sort_idx] == obj_id:
                if use_inverse_weighting:
                    tmp_n_points += inv_data_set[sort_idx]*weight
                else:
                    tmp_n_points += 1.0*weight
    else:
        for pair_id, inv_dist in zip(id_data_set, inv_data_set):
            sort_idx = np.searchsorted(id_array[start_idx:end_idx], pair_id)
            if sort_idx >= end_idx - start_idx or sort_idx < 0:
                continue
            if id_array[start_idx:end_idx][sort_idx] == pair_id:
                weight = weight_array[start_idx:end_idx][sort_idx]
                if use_inverse_weighting:
                    tmp_n_points += inv_dist*weight
                else:
                    tmp_n_points += 1.0*weight
    return tmp_n_points

def _load_pair_data(hdf5_group, key_start, n_load):
    """Functions for loading individual target objects from the HDF5 file.
    ----------------------------------------------------------------------------
    Args:
        hdf5_group: an h5py group object
        key_start: int index of starting object to load
        n_load: int number of objects to load
    Returns:
        list of numpy arrays
    """
    output_list = []
    key_list = hdf5_group.keys()
    key_end = key_start + n_load
    if key_end > len(key_list):
        key_end = len(key_list)
    for key_idx in xrange(key_start, key_end):
        output_list.append([hdf5_group[key_list[key_idx]]['ids'][...],
                            hdf5_group[key_list[key_idx]]['inv_dist'][...]])
    return output_list

def _collapse_full_sample(hdf5_pairs_group, pdf_maker_obj, unknown_data, args):
    """Convience function for collapsing the full sample of ids into a single
    estimate.
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
        unknown_data = unknown_data[unknown_data[args.unknown_weight_name] != 0]
    rand_ratio = (unknown_data.shape[0] /
                  hdf5_pairs_group.attrs['n_random_points'])
    if args.unknown_stomp_region_name is not None:
        tmp_n_region = np.array(
            [unknown_data[unknown_data[args.unknown_stomp_region_name] ==
                          reg_idx].shape[0]
             for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])],
            dtype=np.int_)
        rand_ratio = ((tmp_n_region /
                       hdf5_pairs_group.attrs['n_random_points']) *
                      (hdf5_pairs_group.attrs['area'] /
                       hdf5_pairs_group.attrs['region_area']))
    id_array = unknown_data[args.unknown_index_name]
    id_args_array = id_array.argsort()
    id_array = id_array[id_args_array]
    n_target = len(hdf5_pairs_group)
    target_unknown_array = np.empty(n_target, dtype=np.float32)
    print("\t\tcomputing/storing pair count...")
    for target_idx, key_name in enumerate(hdf5_pairs_group.keys()):
        target_grp = hdf5_pairs_group[key_name]
        if args.use_inverse_weighting:
            target_unknown_array[target_idx] = np.sum(
                target_grp['inv_dist'][...])
        else:
            target_unknown_array[target_idx] = (
                1.*target_grp['ids'][...]).shape[0]
    pdf_maker_obj.set_target_unknown_array(target_unknown_array)
    pdf_maker_obj.scale_random_points(rand_ratio, 1.0)
    return None


class PDFMaker(object):
    """Main class for the heavy lifting of matching an array of object indices
    to the pair hdf5 data file, masking the used/un-used objects, summing the
    data into the spec-z bins, and outputting the posterier redshift
    distribution.
    """
    def __init__(self, hdf5_pair_group, args):
        """Init function for the PDF maker. The init class is a container for
        arrays of single point estimaties around target, known redshift objects.
        The class also computes the estimates of clustering redshift recovery
        in spatial regions and the collapsed single PDF.
        ------------------------------------------------------------------------
        Args:
            hdf5_pair_group: HDF5 group object containing the target object
                pairs
            args: ArgumentParser.parse_args object from
                input_flags.parse_input_pdf_args
        """
        self.target_redshift_array = np.empty(len(hdf5_pair_group),
                                              dtype=np.float32)
        self.target_area_array = np.empty(len(hdf5_pair_group),
                                          dtype=np.float32)
        self.target_density_array = np.empty(len(hdf5_pair_group),
                                             dtype=np.float32)
        self.target_unknown_array = np.empty(len(hdf5_pair_group),
                                             dtype=np.float32)
        self.target_hold_rand_array = np.empty(len(hdf5_pair_group),
                                               dtype=np.float32)
        self.target_region_array = np.empty(len(hdf5_pair_group),
                                            dtype=np.uint32)
        self.target_resolution_array = np.empty(len(hdf5_pair_group),
                                                dtype=np.uint32)
        self._load_data_from_hdf5(hdf5_pair_group, args)
        self._use_target_densities = args.use_reference_cleaning
        self._target_unknown_array_set = False
        self._computed_region_densities = False
        self._computed_pdf = False
        self._computed_bootstraps = False

    def reset_pairs(self):
        self.target_unknown_array = np.zeros_like(self.target_redshift_array)
        self.target_rand_array = np.zeros_like(self.target_redshift_array)
        self._target_unknown_array_set = False
        self._computed_region_densities = False
        self._computed_pdf = False
        self._computed_bootstraps = False
        return None

    def _load_data_from_hdf5(self, hdf5_pair_group, args):
        """Internal function for loading in non-pair search variables such as
        the target redshift, area, region, etc.
        ------------------------------------------------------------------------
        Args:
            hdf5_pair_group: HDF5 group object containing the target object
                pairs
            args: ArgumentParser.parse_args object from
                input_flags.parse_input_pdf_args
        Returns:
            None
        """
        for target_idx, key_name in enumerate(hdf5_pair_group.keys()):
            target_grp = hdf5_pair_group[key_name]
            self.target_redshift_array[target_idx] = (
                target_grp.attrs['redshift'])
            self.target_area_array[target_idx] = target_grp.attrs['area']
            self.target_density_array[target_idx] = target_grp.attrs[
                'target_density']
            self.target_region_array[target_idx] = target_grp.attrs['region']
            self.target_resolution_array[target_idx] = (
                target_grp.attrs['bin_resolution'])
            if args.use_inverse_weighting:
                self.target_hold_rand_array[target_idx] = (
                    target_grp.attrs['rand_inv_dist'])
            else:
                self.target_hold_rand_array[target_idx] = (
                    target_grp.attrs['rand'])

        has_density_mask = np.logical_and(
            self.target_density_array > 0,
            np.isfinite(self.target_density_array))
        min_target_density = self.target_density_array[has_density_mask].min()
        self.target_density_array[
            np.logical_not(has_density_mask)] = min_target_density

        max_n_regions = self.target_region_array.max() + 1
        region_list = []
        for region_idx in xrange(max_n_regions):
            if np.any(region_idx == self.target_region_array):
                region_list.append(region_idx)
        self.region_array = np.array(region_list, dtype=np.uint32)
        self.region_dict = {}
        for array_idx, region_idx in enumerate(self.region_array):
            self.region_dict[region_idx] = array_idx
        return None

    def set_target_unknown_array(self, unknown_array):
        """Function for setting the values of the unknown object density. This
        is done externally rather than internally as Python classes don't play
        to well with the multiprocessing or numba modules.
        ------------------------------------------------------------------------
        Args:
            unknown_array: float array of values defining the number of
                (un)weighted points around a target objet.
        Returns:
            None
        """
        self.target_unknown_array = unknown_array
        self._target_unknown_array_set = True
        return None

    def scale_random_points(self, rand_ratio, ave_weight):
        """Method for setting the scaling relative to the real for the randoms.
        ------------------------------------------------------------------------
        Args:
            rand_ratio: float ratio between the data and randoms
                (# data / # randoms)
            ave_weight: float average value of the weights applied to the
                unknown sample
        Returns:
            None
        """
        # TODO:
        #     Figure out a way to make this more stable.
        try:
            tmp_rand_ratio = rand_ratio[self.target_region_array]
        except IndexError:
            tmp_rand_ratio = rand_ratio
        except TypeError:
            tmp_rand_ratio = rand_ratio
        try:
            tmp_ave_weight = ave_weight[self.target_region_array]
        except IndexError:
            tmp_ave_weight = ave_weight
        except TypeError:
            tmp_ave_weight = ave_weight
        self.target_rand_array = (self.target_hold_rand_array*tmp_rand_ratio *
                                  tmp_ave_weight)
        return None

    def write_target_n_points(self, hdf5_file):
        """Method for writing the intermediate products of the over-density of
        the requested sample per known, target object. This must be run after a
        call to self.colapse_ids_to_single_estimate.
        ------------------------------------------------------------------------
        Args:
            hdf5_file: an open hdf5 object from the return of h5py.File
        Returns:
            None
        """
        if not self._target_unknown_array_set:
            print("PDFMaker.set_target_unknown_array not set. Exiting method.")
            return None
        # TODO
        pass

    def compute_region_densities(self, z_bin_edge_array, z_max):
        """Method for computing the over-density of the unknown sample against
        the target sample binned in target redshift in each of the spatial
        regions of the considered geometry. This allows for spatial
        bootstrapping of the final, resultant PDF. Will not run if
        set_target_unknown_array was not set first.
        ------------------------------------------------------------------------
        Args:
            z_bin_edge_array: float array of the lower bin edge of the redshift
                bins.
            z_max: float maximum redshift of the redshift binning.
        Returns:
            None
        """
        # TODO:
        #     Finish target over-density weighting.
        if not self._target_unknown_array_set:
            print("PDFMaker.set_target_unknown_array not set. Exiting method.")
            return None
        # Initilize arrays.
        self._redshift_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype=np.float32)
        self._n_target_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype=np.uint32)
        self._unknown_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype=np.float32)
        self._rand_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype=np.float32)
        self._area_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype=np.float32)
        self._resolution_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype=np.uint)
        # Loop over target objects
        for target_idx, redshift in enumerate(self.target_redshift_array):
            # If the target object is out of the redshift range continue.
            if redshift < z_bin_edge_array[0] or redshift >= z_max:
                continue
            # Grap the target object region.
            region_idx = self.region_dict[self.target_region_array[target_idx]]
            # Find the redshift bin this object belongs to.
            bin_idx = np.searchsorted(z_bin_edge_array, redshift, 'right') - 1
            # Store object properties.
            self._redshift_reg_array[bin_idx, region_idx] += redshift
            self._n_target_reg_array[bin_idx, region_idx] += 1
            if self._use_target_densities:
                self._unknown_reg_array[bin_idx, region_idx] += (
                    self.target_unknown_array[target_idx] /
                    self.target_density_array[target_idx])
                self._rand_reg_array[bin_idx, region_idx] += (
                    self.target_rand_array[target_idx] /
                    self.target_density_array[target_idx])
            else:
                self._unknown_reg_array[bin_idx, region_idx] += (
                    self.target_unknown_array[target_idx])
                self._rand_reg_array[bin_idx, region_idx] += (
                    self.target_rand_array[target_idx])
            self._area_reg_array[bin_idx, region_idx] += (
                self.target_area_array[target_idx])
            self._resolution_reg_array[bin_idx, region_idx] += (
                self.target_resolution_array[target_idx])
        self._computed_region_densities = True
        return None

    def write_region_densities(self, output_pickle_file, args):
        """Method to write all internal variables describing the over-density
        per spatial region to a pickle file. The data is pickled as a Python
        dictionary.
        ------------------------------------------------------------------------
        Args:
            output_pickle_file: string name of the pickle file to to write out
                to.
        Returns:
            None
        """
        if not self._computed_region_densities:
            print("PDFMaker.compute_region_densities not run. Exiting method.")
            return None
        output_file = open(output_pickle_file, 'w')
        output_dict = {"input_flags": args,
                       "n_regions": self.region_array.shape[0],
                       "redshift": self._redshift_reg_array,
                       "n_target": self._n_target_reg_array,
                       "unknown": self._unknown_reg_array,
                       "rand": self._rand_reg_array,
                       "area": self._area_reg_array,
                       "resolution": self._resolution_reg_array}
        pickle.dump(output_dict, output_file)
        output_file.close()
        return None

    def compute_pdf(self):
        """Method for estimating the redshit posterior distribution of the
        unknown sample without considering the spatial regions. The returned
        over-density vs redshift is calculated using the natural estimator of
        over-density. (DD / DR - 1). Errors are simple Poisson.
        ------------------------------------------------------------------------
        Args:
            None
        Returns:
            None
        """
        if not self._computed_region_densities:
            print("PDFMaker.compute_region_densities not run. Exiting method.")
            return None
        self.redshift_array = (self._redshift_reg_array.sum(axis=1) /
                               self._n_target_reg_array.sum(axis=1))
        self.density_array = (self._unknown_reg_array.sum(axis=1) /
                              self._rand_reg_array.sum(axis=1) - 1.0)
        self.density_err_array = (np.sqrt(self._unknown_reg_array.sum(axis=1)) /
                                  self._rand_reg_array.sum(axis=1))
        self.n_target_array = self._n_target_reg_array.sum(axis=1)
        self.unknown_array = self._unknown_reg_array.sum(axis=1)
        self.rand_array = self._rand_reg_array.sum(axis=1)
        self.area_array = self._area_reg_array.sum(axis=1)
        self.resolution_array = (self._resolution_reg_array.sum(axis=1) /
                                 self._n_target_reg_array.sum(axis=1))
        self._computed_pdf = True
        return None

    def compute_pdf_bootstrap(self, n_bootstraps):
        """Similar to compute_pdf but now the region information is used to
        spatially bootstrap the results in order to estimate errors.
        ------------------------------------------------------------------------
        Args:
            n_bootstraps: int number of spatial bootstraps to sample from the
                regions.
        Returns:
            None
        """
        if not self._computed_region_densities:
            print("PDFMaker.compute_region_densities not run. Exiting method.")
            return None
        self.bootstrap_regions = np.random.randint(
            self.region_array.shape[0],
            size=(n_bootstraps, self.region_array.shape[0]))
        self._compute_pdf_bootstrap(self.bootstrap_regions)
        return None

    def _compute_pdf_bootstrap(self, boot_region_array):
        """Work horse method for computing the bootstrap errors. This method
        takes in an array of bootstrap samples specified by row-wise arrays of
        region ids. Allows for computation of bootstrap errors using the same
        fixed bootstrap samples.
        ------------------------------------------------------------------------
        Args:
            boot_region_array: array of integer region ids
        Returns:
            None
        """
        if not self._computed_region_densities:
            print("PDFMaker.compute_region_densities not run. Exiting method.")
            return None
        self.bootstrap_array = np.empty((self._redshift_reg_array.shape[0],
                                         boot_region_array.shape[0]))
        for boot_idx, boot_regions in enumerate(boot_region_array):
            self.bootstrap_array[:, boot_idx] = np.where(
                self._rand_reg_array[:, boot_regions].sum(axis=1) > 0,
                self._unknown_reg_array[:, boot_regions].sum(axis=1) /
                self._rand_reg_array[:, boot_regions].sum(axis=1) - 1.0,
                0.0)
        self.redshift_array = (self._redshift_reg_array.sum(axis=1) /
                               self._n_target_reg_array.sum(axis=1))
        self.density_array = np.nanmean(self.bootstrap_array, axis=1)
        self.density_err_array = np.nanstd(self.bootstrap_array, axis=1)
        self.n_target_array = self._n_target_reg_array.sum(axis=1)
        self.unknown_array = self._unknown_reg_array.sum(axis=1)
        self.rand_array = self._rand_reg_array.sum(axis=1)
        self.area_array = self._area_reg_array.sum(axis=1)
        self.resolution_array = (self._resolution_reg_array.sum(axis=1) /
                                 self._n_target_reg_array.sum(axis=1))
        self._computed_pdf = True
        self._computed_bootstraps = True
        return None

    def write_bootstrap_samples_to_ascii(self, output_name, args):
        """Method for writing the individual bootstrap samples to ascii.
        ------------------------------------------------------------------------
        Args:
            output_name: string specifying the name of the ascii file to write
                the pdf/density results to. By default any existing file will
                be overwritten.
            args: ArgumentParser.parse_args object returned from
                input_flags.parse_input_pdf_args
        Returns:
            None
        """
        output_header = '# input_flags:\n'
        for arg in vars(args):
            output_header += '#\t%s : %s\n' % (arg, getattr(args, arg))
        np.savetxt(output_name, self.bootstrap_array, fmt='%.8f',
                   header=output_header)
        return None

    def write_pdf_to_ascii(self, output_file):
        """Method for writing the results of the different compute pdf methods
        to ascii.
        ------------------------------------------------------------------------
        Args:
            output_name: Python file object specifying the ascii file to write
                the pdf/density results to. By default any existing file will
                be overwritten.
        Returns:
            None
        """
        if not self._computed_pdf:
            print("PDFMaker.compute_pdf or PDFMaker.compute_pdf_bootstrap not "
                  "run. Exiting method.")
            return None
        output_file.writelines('#type1 = redshift\n')
        output_file.writelines('#type2 = over_density\n')
        output_file.writelines('#type3 = over_density_err\n')
        output_file.writelines('#type4 = n_points\n')
        output_file.writelines('#type5 = n_random\n')
        output_file.writelines('#type6 = area\n')
        output_file.writelines('#type7 = ave resolution\n')
        for bin_idx in xrange(self.redshift_array.shape[0]):
            output_file.writelines(
                '%.8e %.8e %.8e %.8e %.8e %.8e %.8e\n' %
                (self.redshift_array[bin_idx], self.density_array[bin_idx],
                 self.density_err_array[bin_idx], self.unknown_array[bin_idx],
                 self.rand_array[bin_idx], self.area_array[bin_idx],
                 self.resolution_array[bin_idx]))
        return None
