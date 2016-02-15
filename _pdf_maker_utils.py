
import __core__
import _input_flags
import h5py
import numpy as np
import sys

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
    
    comov_min = __core__.Planck13.comoving_distance(z_min)
    comov_max = __core__.Planck13.comoving_distance(z_max)
    return __core__.redshift(np.arange(comov_min, comov_max,
                                       (comov_max - comov_min) / (1. * n_bins)))


class PDFMaker(object):
    
    """
    Main class for the heavy lifting of matching an array of object indices to
    the pair hdf5 data file, masking the used/un-used objects, summing the data
    into the spec-z bins, and outputting the posterier redshift distribution. 
    """
    
    def __init__(self, hdf5_file):
        
        """
        Init function for the PDF maker. The init method takes in an hdf5 file
        object containing the pairs of spectra to photometric objects.
        Args:
            hdf5_file: h5py object returned by h5py.File()
        Returns:
            None
        """
        
        self._hdf5_pairs = hdf5_file
        
    def colapse_ids_to_single_estimate(self, scale_name, id_array):
        
        """
        This the main functionallity of the class. It enables the matching of
        a set of catalog ids to the ids stored as pairs to the spectroscopic
        objects. The result of this calculation is a intermediary data product
        containing the density of unknown objects around each target object.
        Args:
            scale_name: string specifying the name of the scale to load from the
        pair hdf5 file
            id_array: numpy.array of ints specifying the catalog ids of the
        unknown objects. WARNING: make sure that these ids are cut to the
        geometry as the target, known objects. At best you'll waste a lot of
        time searching for objects that don't overlap, at worst you'll get no
        results.
        Returns:
            None (Returns are stored within the class)
        """
        
        scale_grp = self._hdf5_pairs[scale_name]
        self._rand_ratio = (id_array.shape[0] /
                            (1. * scale_grp.attrs['n_random_points']))
        
        self._target_redshift_array = np.empty(len(scale_grp))
        self._target_area_array = np.empty(len(scale_grp))
        self._target_n_points_array = np.empty(len(scale_grp),
                                               dtype = np.uint32)
        self._target_n_rand_array = np.empty(len(scale_grp), dtype = np.uint32)
        self._target_region_array = np.empty(len(scale_grp), dtype = np.uint32)
        
        ### TODO:
        ###     This loop will likely be inefficient both in terms of time spent
        ###     and reads from disk. Options are to use numba where posible and
        ###     parallization across multiple cores. For the load times posibly
        ###     set a worker to preload portions from disk. (min 100 MB chunks)
        for target_idx, key_name in enumerate(scale_grp.keys()):
            data_set = scale_grp[key_name]
            self._target_redshift_array[target_idx] = data_set.attrs['redshift']
            self._target_area_array[target_idx] = data_set.attrs['area']
            tmp_n_points = 0
            for obj_id in data_set:
                if np.any(id_array == obj_id): tmp_n_points += 1
            self._target_n_points_array[target_idx] = tmp_n_points
            self._target_n_rand_array[target_idx] = data_set.attrs['rand']
            self._target_region_array[target_idx] = data_set.attrs['region']
            
        max_n_regions = self._target_region_array.max()
        region_list = []
        for region_idx in xrange(max_n_regions):
            if np.any(region_idx == region_idx):
                region_list.append(region_idx)
        self._region_array = np.array(region_list, dtype = np.uint32)
        
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
        
        ### TODO:
        ###     Code this
        
        pass
        
    def compute_pdf(self, z_bin_edge_array, z_max):
        
        """
        Method for estimating the redshit posterior distribution of the unknown
        sample calculated int self.colapse_ids_to_single_estimate. The returned
        over-density vs redshift is calculated using the natural estimator of
        over-density. (DD / DR - 1).
        Args:
            z_bin_edge_array: numpy.array of floats defining the lower edge of 
        the redshift bins
            z_max: float maximum redshift to estimate the to.
        Returns:
            None
        """
        
        self._redshift_array = np.zeros_like(z_bin_edge_array)
        self._n_target_array = np.zeros_like(z_bin_edge_array,dtype = np.int_)
        self._n_points_array = np.zeros_like(z_bin_edge_array, dtype = np.int_)
        self._n_rand_array = np.zeros_like(z_bin_edge_array, dtype = np.int_)
        self._area_array = np.zeros_like(z_bin_edge_array)
    
        for target_idx, redshift in self._target_redshift_array:
            
            if redshift < z_bin_edge_array[0] or redshift >= z_max:
                continue
            bin_idx = np.searchsorted(z_bin_edge_array, redshift, 'right') - 1
            self._n_target_array[bin_idx] += 1
            self._redshift_array[bin_idx] += redshift
            self._n_points_array[bin_idx] += (
                self._target_n_points_array[target_idx])
            self._n_rand_array[bin_idx] += (
                self._target_n_rand_array[target_idx])
            self._area_array[bin_idx] += self._target_area_array[target_idx]
            
        
        self._redshift_array /= 1. * self._n_target_array
        self._density_array = (self._n_points_array /
                               (self._n_rand_array * self._rand_ratio) - 1.0)
        self._density_err_array = 1. / np.sqrt(self._n_points_array)
        
        return None  
        
    def compute_pdf_bootstrap(self, z_bin_edge_array, z_max, n_bootraps,
                              output_raw_bootstraps_name):
        
        self._redshift_array = np.zeros(z_bin_edge_array.shape[0])
        self._n_target_array = np.zeros(
            (self._region_array.shape[0], z_bin_edge_array.shape[0]),
            dtype = np.int_)
        self._n_points_array = np.zeros(
            (self._region_array.shape[0], z_bin_edge_array.shape[0]),
            dtype = np.int_)
        self._n_rand_array = np.zeros(
            (self._region_array.shape[0], z_bin_edge_array.shape[0]),
            dtype = np.int_)
        self._area_array = np.zeros((self._region_array.shape[0],
                                     z_bin_edge_array.shape[0]))
        
        for target_idx, redshift in self._target_redshift_array:
            
            if redshift < z_bin_edge_array[0] or redshift >= z_max:
                continue
            bin_idx = np.searchsorted(z_bin_edge_array, redshift, 'right') - 1
            region_idx = np.where(self._target_region_array[target_idx] ==
                                  self._region_array)
            self._n_target_array[region_idx, bin_idx] += 1
            self._redshift_array[region_idx, bin_idx] += redshift
            self._n_points_array[region_idx, bin_idx] += (
                self._target_n_points_array[target_idx])
            self._n_rand_array[region_idx, bin_idx] += (
                self._target_n_rand_array[target_idx])
            self._area_array[region_idx, bin_idx] += (
                self._target_area_array[target_idx])
            
        self._redshift_array /= 1. * self._n_target_array.sum(axis = 0)
        self._density_bootstraps = None
    
    def straight_sum(self, scale_name, z_bin_edge_array, z_max):
        
        scale_grp = self._hdf5_pairs[scale_name]
        self._rand_ratio = (scale_grp.attrs['n_unknown'] /
                            (1. * scale_grp.attrs['n_random_points']))
        
        self._redshift_array = np.zeros_like(z_bin_edge_array)
        self._n_target_array = np.zeros_like(z_bin_edge_array,dtype = np.int_)
        self._n_points_array = np.zeros_like(z_bin_edge_array, dtype = np.int_)
        self._n_rand_array = np.zeros_like(z_bin_edge_array, dtype = np.int_)
        self._area_array = np.zeros_like(z_bin_edge_array)
        
        for key_name in scale_grp.keys():
            data_set = scale_grp[key_name]
            redshift = data_set.attrs['redshift']
            if redshift < z_bin_edge_array[0] or redshift >= z_max:
                continue
            bin_idx = np.searchsorted(z_bin_edge_array, redshift, 'right') - 1
            self._n_target_array[bin_idx] += 1
            self._redshift_array[bin_idx] += redshift
            self._n_points_array[bin_idx] += data_set.shape[0]
            self._n_rand_array[bin_idx] += data_set.attrs['rand']
            self._area_array[bin_idx] += data_set.attrs['area']

        self._redshift_array /= 1. * self._n_target_array
        self._density_array = (self._n_points_array /
                               (self._n_rand_array * self._rand_ratio) - 1.0)
        self._density_err_array = 1. / np.sqrt(self._n_points_array)
        
        return None
    
    def write_to_ascii(self, output_name):
        
        output_file = open(output_name, 'w')
        
        output_file.writelines('#type1 = redshift\n')
        output_file.writelines('#type2 = over_density\n')
        output_file.writelines('#type3 = over_density_err\n')
        output_file.writelines('#type4 = n_points\n')
        output_file.writelines('#type5 = n_random\n')
        output_file.writelines('#type6 = area\n')
        for bin_idx in xrange(self._redshift_array.shape[0]):
            
            output_file.writelines(
                '%.6f %.6f %.6f %i %i %.6f\n' %
                (self._redshift_array[bin_idx], self._density_array[bin_idx],
                 self._density_err_array[bin_idx],
                 self._n_points_array[bin_idx],
                 self._n_rand_array[bin_idx] * self._rand_ratio,
                 self._area_array[bin_idx]))
        output_file.close()
        
        return None
    
    