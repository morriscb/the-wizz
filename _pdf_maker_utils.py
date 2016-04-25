
import _core_utils
from copy import copy
import input_flags
import h5py
from multiprocessing import Array, Pool
import numpy as np
import pickle
import sys
from _core_utils import redshift
from __builtin__ import True

def _create_linear_redshift_bin_edges(z_min, z_max, n_bins):
    
    """
    Simple utility for computing redshift bins that are linearly spaced in 
    redshift. Not recommened for use if your concern is maximum signal to noise.
    args:
        z_min: float, minimum redshift to bin from
        z_max: float, maximum redshift to bin to
        n_bins: int, number of bins
    returns:
        numpy.array of type float and shape (n_bins,) containing the lower bin
        edges. The n_bin + 1 edge is equal to z_max.
    """
        
    return np.arange(z_min, z_max, (z_max - z_min) / (1. * n_bins))

def _create_adaptive_redshift_bin_edges(z_min, z_max, n_bins, redshift_array):
    
    """
    Simple utility for computing redshift bins that delivers a consistent number
    of spectroscopic objects per bin. 
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
                                     useable_z_array.shape[0] / (1.0 * n_bins),
                                     dtype = np.int_)]
    
def _create_comoving_redshift_bin_edges(z_min, z_max, n_bins):
    
    """
    Simple utility for computing redshift bins that are equally spaced in
    comoving, line of sight distance. This creates bins that have a smoother
    bias versus redshift.
    args:
        z_min: float, minimum redshift to bin from
        z_max: float, maximum redshift to bin to
        n_bins: int, number of bins
    returns:
        numpy.array of type float and shape (n_bins,) containing the lower bin
        edges. The n_bin + 1 edge is equal to z_max.
    """
    
    comov_min = _core_utils.Planck13.comoving_distance(z_min).value
    comov_max = _core_utils.Planck13.comoving_distance(z_max).value
    return _core_utils.redshift(
        np.arange(comov_min, comov_max,
                  (comov_max - comov_min) / (1. * n_bins)))

def collapse_ids_to_single_estimate(hdf5_pairs_group, pdf_maker_obj,
                                    unknown_data, args):
        
    """
    This is the heart of The-wiZZ. It enables the matching of
    a set of catalog ids to the ids stored as pairs to the spectroscopic
    objects. The result of this calculation is a intermediary data product
    containing the density of unknown objects around each target object stored
    in the PDFMaker data structure class.
    Args:
        hdf5_pairs_group: hdf5 group object containing the pair ids for a fixed
            annulus.
        unknown_data: open fits data containing object ids and relivent weights
        args: ArgumentParser.parse_args object returned from
            input_flags.parse_input_pdf_args
    Returns:
        PDFMaker class object containing the resultant over densities.
    """
    
    print("\tpre-loading unknown data...")
    if args.unknown_weight_name is not None:
        unknown_data = unknown_data[unknown_data[args.weight_name] != 0]
    rand_ratio = (unknown_data.shape[0] /
                  (1. * hdf5_pairs_group.attrs['n_random_points']))
    if args.unknown_stomp_region_name is not None:
        tmp_n_region = np.array(
            [unknown_data[unknown_data[args.unknown_stomp_region_name] ==
                          reg_idx].shape[0]
             for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])],
                                   dtype = np.int_)
        rand_ratio = (
            (tmp_n_region / (1. * hdf5_pairs_group.attrs['n_random_points'])) *
            (hdf5_pairs_group.attrs['area'] /
             hdf5_pairs_group.attrs['region_area']))
    id_array = unknown_data[args.unknown_index_name]
    id_args_array = id_array.argsort()
    id_array = id_array[id_args_array]
    
    ave_weight = 1.0
    weight_array = np.ones(unknown_data.shape[0], dtype = np.float32)
    if args.unknown_weight_name is not None:
        weight_array = unknown_data[args.unknown_weight_name][id_args_array]
        ave_weight = np.mean(weight_array)
        if args.unknown_stomp_region_name is not None:
            ave_weight = np.array(
                [unknown_data[args.unknown_weight_name][
                     unknown_data[args.unknown_stomp_region_name] ==
                     reg_idx].mean()
                 for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])],
                                       dtype = np.float_)
    
    n_target = len(hdf5_pairs_group)
    target_unknown_array = np.empty(n_target, dtype = np.float32)
    
    pair_start = 0
    hold_pair_start = 0
    pair_data = []
    pool = Pool(args.n_processes)
    while hold_pair_start < n_target:
   
        ### TODO:
        ###     Make the multiprocessing better. Currently the code copies over
        ###     the full information of id_array and weight array to the child
        ###     processes. This will be bad when these arrays become large.
        if len(pair_data) > 0:
            print("\t\tmatching pairs: starting targets %i-%i..." %
                  (hold_pair_start, hold_pair_start + args.n_target_load_size))
            pool_iter = pool.imap(
                _collapse_multiplex,
                [(data_set, id_array, weight_array, args.use_inverse_weighting)
                 for pair_idx, data_set in enumerate(pair_data)],
                chunksize = np.int(np.where(
                    args.n_processes > 1,
                    np.log(len(pair_data)) / np.log(args.n_processes), 1)))
        
        print("\t\tloading next pairs...")
        pair_data = _load_pair_data(hdf5_pairs_group, pair_start,
                                    args.n_target_load_size)
        
        try:
            type(pool_iter)
            print("\t\tcomputing/storing pair count...")
            for pair_idx, target_value in enumerate(pool_iter):
                target_unknown_array[hold_pair_start + pair_idx] = target_value
        except UnboundLocalError:
            pass
        
        hold_pair_start = pair_start
        pair_start += args.n_target_load_size
        
    pool.close()
    pool.join()
    
    pdf_maker_obj.set_target_unknown_array(target_unknown_array)
    pdf_maker_obj.scale_random_points(rand_ratio, ave_weight)
    
    return None

def _collapse_multiplex(input_tuple):

    (data_set, id_array, weight_array,
     use_inverse_weighting) = input_tuple

    id_data_set, inv_data_set = data_set
    if len(id_data_set) == 0:
        return 0.0
    ### Since the ids around the target are partially localized spatially we
    ### will loop over the unknown ids and match them into the target ids.
    start_idx = np.searchsorted(id_array, id_data_set[0])
    end_idx = np.searchsorted(id_array, id_data_set[-1],
                              side = 'right')
    if start_idx == end_idx:
        return 0.0
    
    if start_idx < 0:
        start_idx = 0
    if end_idx > id_array.shape[0]:
        end_idx = id_array.shape[0]
    
    tmp_n_points = 0.0
    for obj_id, weight in zip(id_array[start_idx:end_idx],
                              weight_array[start_idx:end_idx]):
        sort_idx = np.searchsorted(id_data_set, obj_id)
        if sort_idx >= len(id_data_set) or sort_idx < 0:
            continue
        if id_data_set[sort_idx] == obj_id:
            if use_inverse_weighting:
                tmp_n_points += inv_data_set[sort_idx] * weight
            else:
                tmp_n_points += 1.0 * weight
    
    return tmp_n_points

def _load_pair_data(hdf5_group, key_start, n_load):

    output_list = []
    key_list = hdf5_group.keys()
    key_end = key_start + n_load
    if key_end > len(key_list):
        key_end = len(key_list)
    for key_idx in xrange(key_start, key_end):
        output_list.append([hdf5_group[key_list[key_idx]]['ids'][...],
                            hdf5_group[key_list[key_idx]]['inv_dist'][...]])
    return output_list

class PDFMaker(object):
    
    """
    Main class for the heavy lifting of matching an array of object indices to
    the pair hdf5 data file, masking the used/un-used objects, summing the data
    into the spec-z bins, and outputting the posterier redshift distribution. 
    """
    
    def __init__(self, hdf5_pair_group, args):
        
        """
        Init function for the PDF maker. The init class is a container for
        arrays of single point estimaties around target, known redshift objects.
        The class also computes the estimates of clustering redshift recovery
        in spatial regions and the collapsed single PDF.
        Args:
            hdf5_pair_group: HDF5 group object containing the target object
                pairs
            args: ArgumentParser.parse_args object from
                input_flags.parse_input_pdf_args
        """
        
        self.target_redshift_array = np.empty(len(hdf5_pair_group),
                                              dtype = np.float32)
        self.target_area_array = np.empty(len(hdf5_pair_group),
                                          dtype = np.float32)
        self.target_unknown_array = np.empty(len(hdf5_pair_group),
                                              dtype = np.float32)
        self.target_hold_rand_array = np.empty(len(hdf5_pair_group),
                                               dtype = np.float32)
        self.target_region_array = np.empty(len(hdf5_pair_group),
                                            dtype = np.uint32)
        self.target_resolution_array = np.empty(len(hdf5_pair_group),
                                                dtype = np.uint32)
        
        self._load_data_from_hdf5(hdf5_pair_group, args)
        
        self._target_unknown_array_set = False
        self._computed_region_densities = False
        self._computed_pdf = False
        self._computed_bootstraps = False
        
    def _load_data_from_hdf5(self, hdf5_pair_group, args):
        
        """
        Internal function for loading in non-pair search variables such as the
        target redshift, area, region, etc.
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
            self.target_region_array[target_idx] = target_grp.attrs['region']
            self.target_resolution_array[target_idx] = (
                target_grp.attrs['bin_resolution'])
            if args.use_inverse_weighting:
                self.target_hold_rand_array[target_idx] = (
                    target_grp.attrs['rand_inv_dist'])
            else:
                self.target_hold_rand_array[target_idx] = (
                    target_grp.attrs['rand'])
                
        max_n_regions = self.target_region_array.max() + 1
        region_list = []
        for region_idx in xrange(max_n_regions):
            if np.any(region_idx == self.target_region_array):
                region_list.append(region_idx)
        self.region_array = np.array(region_list, dtype = np.uint32)
        
        self.region_dict = {}
        for array_idx, region_idx in enumerate(self.region_array):
            self.region_dict[region_idx] = array_idx
        
        return None
                
    def set_target_unknown_array(self, unknown_array):
        
        """
        Function for setting the values of the unknown object density. This is
        done externally rather than internally as Python classes don't play to
        well with the multiprocessing or numba modules.
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
        
        """
        Method for setting the scaling relative to the real for the randoms.
        Args:
            rand_ratio: float ratio between the data and randoms
                (# data / # randoms)
            ave_weight: float average value of the weights applied to the
                unknown sample
        Returns:
            None
        """
        
        ### TODO:
        ###     Figure out a way to make this more stable.
        
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
        self.target_rand_array = (self.target_hold_rand_array * tmp_rand_ratio *
                                  tmp_ave_weight)
        
        return None
    
    def write_target_n_points(self, hdf5_file):
        
        """
        Method for writing the intermediate products of the over-density of the
        requested sample per known, target object. This must be run after a call
        to self.colapse_ids_to_single_estimate.
        Args:
            hdf5_file: an open hdf5 object from the return of h5py.File
        Returns:
            None
        """
        
        if not self._target_unknown_array_set:
            print("PDFMaker.set_target_unknown_array not set. Exiting method.")
            return None
        
        ### TODO
        
        pass
    
    def compute_region_densities(self, z_bin_edge_array, z_max):
        
        """
        Method for computing the over-density of the unknown sample against the
        target sample binned in target redshift in each of the spatial regions
        of the considered geometry. This allows for spatial bootstrapping of the
        final, resultant PDF. Will not run if set_target_unknown_array was not
        set first.
        Args:
            z_bin_edge_array: float array of the lower bin edge of the redshift
                bins.
            z_max: float maximum redshift of the redshift binning.
        Returns:
            None
        """
        ### TODO:
        ###     Make a default storage such that regions without certain
        ### redshift are not considered for those bins. This will allow for
        ### combining of different target samples between surveys.
        
        if not self._target_unknown_array_set:
            print("PDFMaker.set_target_unknown_array not set. Exiting method.")
            return None
        
        self._redshift_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype = np.float32)
        self._n_target_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype = np.uint32)
        self._unknown_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype = np.float32)
        self._rand_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype = np.float32)
        self._area_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype = np.float32)
        self._resolution_reg_array = np.zeros(
            (z_bin_edge_array.shape[0], self.region_array.shape[0]),
            dtype = np.uint)
        
        for target_idx, redshift in enumerate(self.target_redshift_array):
            
            if redshift < z_bin_edge_array[0] or redshift >= z_max:
                continue
            region_idx = self.region_dict[self.target_region_array[target_idx]]
            bin_idx = np.searchsorted(z_bin_edge_array, redshift, 'right') - 1
            self._redshift_reg_array[bin_idx, region_idx] += redshift
            self._n_target_reg_array[bin_idx, region_idx] += 1
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
        
        """
        Method to write all internal variables describing the over-density per
        spatial region to a pickle file. The data is pickled as a Python
        dictionary.
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
        output_dict = {"input_flags" : args,
                       "n_regions" : self.region_array.shape[0],
                       "redshift" : self._redshift_reg_array,
                       "n_target" : self._n_target_reg_array,
                       "unknown" : self._unknown_reg_array,
                       "rand" : self._rand_reg_array,
                       "area" : self._area_reg_array,
                       "resolution" : self._resolution_reg_array}
        pickle.dump(output_dict, output_file)
        output_file.close()
        
        return None
        
    def compute_pdf(self):
        
        """
        Method for estimating the redshit posterior distribution of the unknown
        sample without considering the spatial regions. The returned
        over-density vs redshift is calculated using the natural estimator of
        over-density. (DD / DR - 1). Errors are simple Poisson.
        Args:
            None
        Returns:
            None
        """
        
        if not self._computed_region_densities:
            print("PDFMaker.compute_region_densities not run. Exiting method.")
            return None
        
        self.redshift_array = (self._redshift_reg_array.sum(axis = 1) /
                               self._n_target_reg_array.sum(axis = 1))
        self.density_array = (self._unknown_reg_array.sum(axis = 1) /
                              self._rand_reg_array.sum(axis = 1) - 1.0)
        self.density_err_array = (
            np.sqrt(self._unknown_reg_array.sum(axis = 1)) /
            self._rand_reg_array.sum(axis = 1))
        self.n_target_array = self._n_target_reg_array.sum(axis = 1)
        self.unknown_array = self._unknown_reg_array.sum(axis = 1)
        self.rand_array = self._rand_reg_array.sum(axis = 1)
        self.area_array = self._area_reg_array.sum(axis = 1)
        self.resolution_array = (self._resolution_reg_array.sum(axis = 1) /
                                 (1. * self._n_target_reg_array.sum(axis = 1)))
        
        self._computed_pdf = True
        
        return None
    
    def compute_pdf_bootstrap(self, n_bootstraps):
        """
        Similar to compute_pdf but now the region information is used to
        spatially bootstrap the results in order to estimate errors.
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
            size = (n_bootstraps, self.region_array.shape[0]))
        self._compute_pdf_bootstrap(self.bootstrap_regions)
        
        return None
        
    def _compute_pdf_bootstrap(self, boot_region_array):
        
        """
        Work horse method for computing the bootstrap errors. This method takes
        in an array of bootstrap samples specified by row-wise arrays of region
        ids. Allows for computation of bootstrap errors using the same fixed
        bootstrap samples. 
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
                self._rand_reg_array[:, boot_regions].sum(axis = 1) > 0,     
                self._unknown_reg_array[:, boot_regions].sum(axis = 1) /
                self._rand_reg_array[:, boot_regions].sum(axis = 1) - 1.0,
                0.0)
            
        self.redshift_array = (self._redshift_reg_array.sum(axis = 1) /
                               self._n_target_reg_array.sum(axis = 1))
        self.density_array = np.nanmean(self.bootstrap_array, axis = 1)
        self.density_err_array = np.nanstd(self.bootstrap_array, axis = 1)
        self.n_target_array = self._n_target_reg_array.sum(axis = 1)
        self.unknown_array = self._unknown_reg_array.sum(axis = 1)
        self.rand_array = self._rand_reg_array.sum(axis = 1)
        self.area_array = self._area_reg_array.sum(axis = 1)
        self.resolution_array = (self._resolution_reg_array.sum(axis = 1) /
                                 (1. * self._n_target_reg_array.sum(axis = 1)))
        
        self._computed_pdf = True
        self._computed_bootstraps = True
        
        return None
    
    def write_bootstrap_samples_to_ascii(self, output_name, args):
        
        """
        Method for writing the individual bootstrap samples to ascii.
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
        
        np.savetxt(output_name, self.bootstrap_array, fmt = '%.8f',
                   header = output_header)
        
        return None
    
    def write_pdf_to_ascii(self, output_file):
        
        """
        Method for writing the results of the different compute pdf methods to
        ascii.
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
    
    