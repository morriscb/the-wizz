
import _core_utils
import input_flags
import h5py
from multiprocessing import Pool
import numpy as np
import sys
from _core_utils import redshift

### TODO:
###     Restructure code for multiprocessing/numba. This will require changing
### the class structure to more of a container that gets acted on by functions
### in this file. This should be a feature of the code by the alpha state.

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
    
    comov_min = _core_utils.Planck13.comoving_distance(z_min)
    comov_max = _core_utils.Planck13.comoving_distance(z_max)
    return _core_utils.redshift(np.arange(comov_min, comov_max,
                                       (comov_max - comov_min) / (1. * n_bins)))

def collapse_ids_to_single_estimate(hdf5_pairs_group, unknown_data, args):
        
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
    
    rand_ratio = (unknown_data.shape[0] /
                  (1. * hdf5_pairs_group.attrs['n_random_points']))
    id_array = unknown_data[args.unknown_index_name]
    id_args_array = id_array.argsort()
    id_array = id_args_array[id_args_array]
    ave_weight = 1.0
    if args.unknown_weight_name is not None:
        weight_array = unknown_data[args.unknown_weight_name][id_args_array]
        ave_weight = weight_array.mean()
    
    pdf_maker = PDFMaker(hdf5_pairs_group, args)
    
    target_unknown_array = np.empty(len(hdf5_pairs_group), dtype = np.float32)
    
    for target_idx, key_name in enumerate(hdf5_pairs_group.keys()):
        target_group = hdf5_pairs_group[key_name]
        data_set = target_group['ids'][...]
        invdata_set = target_group['inv_dist'][...]
        tmp_n_points = 0
        for obj_id, inv_weight in zip(data_set, invdata_set):
            sort_idx = np.searchsorted(id_array, obj_id)
            if id_array[sort_idx] == obj_id:
                if args.unknown_weight_name is None:
                    weight = 1.0
                else:
                    weight = weight_array[sort_idx]
                if args.use_inverse_weighting:
                    tmp_n_points += inv_weight * weight
                else:
                    tmp_n_points += 1.0 * weight
        target_unknown_array[target_idx] = tmp_n_points
        
    pdf_maker.set_target_unknown_array(target_unknown_array)
    pdf_maker.scale_random_points(rand_ratio, ave_weight)
    
    return pdf_maker

def _collapsed_multiplex():
    
    pass

def _collapsed_numba():
    
    pass


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
        self.target_rand_array = np.empty(len(hdf5_pair_group),
                                          dtype = np.float32)
        self.target_region_array = np.empty(len(hdf5_pair_group),
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
            self.target_redshift_array[target_idx] = target_grp.attrs['redshift']
            self.target_area_array[target_idx] = target_grp.attrs['area']
            self.target_region_array[target_idx] = target_grp.attrs['region']
            if args.use_inverse_weighting:
                self.target_rand_array[target_idx] = (
                    target_grp.attrs['rand_invdist'])
            else:
                self.target_rand_array = target_grp.attrs['rand']
                
        max_n_regions = self.target_region_array.max()
        region_list = []
        for region_idx in xrange(max_n_regions):
            if np.any(region_idx == region_idx):
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
        
        self.target_rand_array *= rand_ratio * ave_weight
        
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
        
        for target_idx, redshift in enumerate(self._target_redshift_array):
            
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
            
        return None
    
    def write_region_densities(self, output_pickle_file):
        
        """
        Method to write all internal variables describing the over-density per
        spatial region to a pickle file.
        Args:
            output_pickle_file: string name of the pickle file to to write out
                to.
        Returns:
            None
        """
        
        if not self._computed_region_densities:
            print("PDFMaker.compute_region_densities not run. Exiting method.")
            return None
        
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
            self._rand_array.axis(axis = 1))
        self.n_target_array = self._n_target_reg_array.sum(axis = 1)
        self.unknown_array = self._unknown_reg_array.sum(axis = 1)
        self.rand_array = self._rand_reg_array.sum(axis = 1)
        self.area_array = self._area_reg_array.sum(axis = 1)
        
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
        
        self.bootstrap_array = np.empty((self._redshift_reg_array.shape[1],
                                         n_bootstraps))
        
        for boot_idx in xrange(n_bootstraps):
            boot_regions = np.random.randint(self.region_array.shape[0],
                                             size = self.region_array.shape[0])
            self.bootstrap_array[:, boot_idx] = (
                self._unknown_reg_array[:, boot_regions].sum(axis = 1) /
                self._rand_reg_array[:, boot_regions].sum(axis = 1) - 1.0)
            
        self.redshift_array = (self._redshift_reg_array.sum(axis = 1) /
                               self._n_target_reg_array.sum(axis = 1))
        self.density_array = self.bootstrap_array.mean(axis = 1)
        self.density_array_err = self.bootstrap_array.std(axis = 1)
        self.n_target_array = self._n_target_reg_array.sum(axis = 1)
        self.unknown_array = self._unknown_reg_array.sum(axis = 1)
        self.rand_array = self._rand_reg_array.sum(axis = 1)
        self.area_array = self._area_reg_array.sum(axis = 1)
            
        return None
    
    def write_bootstrap_samples_to_ascii(self, output_name):
        
        """
        Method for writing the individual bootstrap samples to ascii.
        Args:
            output_name: string specifying the name of the ascii file to write
                the pdf/density results to. By default any existing file will
                be overwritten.
        Returns:
            None
        """
        
        np.savetxt(output_name, self.bootstrap_array)
        
        return None
    
    def write_pdf_to_ascii(self, output_name):
        
        """
        Method for writing the results of the different compute pdf methods to
        ascii.
        Args:
            output_name: string specifying the name of the ascii file to write
                the pdf/density results to. By default any existing file will
                be overwritten.
        Returns:
            None
        """
        
        if not self._computed_pdf:
            print("PDFMaker.compute_pdf or PDFMaker.compute_pdf_bootstrap not "
                  "run. Exiting method.")
            return None
            
        output_file = open(output_name, 'w')
        
        output_file.writelines('#type1 = redshift\n')
        output_file.writelines('#type2 = over_density\n')
        output_file.writelines('#type3 = over_density_err\n')
        output_file.writelines('#type4 = n_points\n')
        output_file.writelines('#type5 = n_random\n')
        output_file.writelines('#type6 = area\n')
        for bin_idx in xrange(self._redshift_array.shape[0]):
            
            output_file.writelines(
                '%.6e %.6e %.6e %.6e %.6e %.6e\n' %
                (self.redshift_array[bin_idx], self.density_array[bin_idx],
                 self.density_err_array[bin_idx], self.unknown_array[bin_idx],
                 self.rand_array[bin_idx], self.area_array[bin_idx]))
            
        output_file.close()
        
        return None
    
    