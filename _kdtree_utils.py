
import numpy as np
from scipy.spatial import cKDTree

### TODO:
###     Pickling doesn't work with either KDTree object. Need to think of a
###     better way of loading and storing the nearest neighbors.



def create_match_data(input_catalog, mag_name_list, other_name_list,
                      use_as_colors):
    
    kdtree_data_array = np.empty((input_catalog.shape[0], len(mag_name_list)))
    
    for mag_idx, mag_name in enumerate(mag_name_list):
        
        kdtree_data_array[:,mag_idx] = input_catalog[mag_name]
        if mag_idx > 0 and use_as_colors:
            kdtree_data_array[:,mag_idx - 1] -= kdtree_data_array[:,mag_idx]
    if use_as_colors:
        kdtree_data_array = np.delete(kdtree_data_array, -1, 1)
        
    if len(other_name_list) > 0:
        other_data_array = np.empty((input_catalog.shape[0],
                                     len(other_name_list)))
        for other_idx, other_name in enumerate(other_name_list):
            
            other_data_array[:,other_idx] = input_catalog[other_name]
        
        kdtree_data_array = np.concatenate(
            (kdtree_data_array, other_data_array), axis = 1)
    
    return kdtree_data_array


class CatalogKDTree(object):

    """
    Convience class for creating a dataset suitable for a KDTree search, and
    wrapping the scipy KDTree object. 
    """
    
    def __init__(self, input_array):
        
        """
        __init__ method preps the internal data storage and creates the KDTree.
        Args:
            input_catalog: astropy.io.fits catalog object containing the columns
                of interest
            column_name_list: list of string names of catalog columns to
               consider for the KDTree
            id_column_name: string name of the column containing the indices
        """
        
        self._internal_array = input_array
            
        self._normalize_data()
        self._initialize_tree()
    
    def __call__(self, input_array, k):
        
        """
        Given input properties of an object, return the KDTree, array indices of
        the k nearest neighbors.
        Args:
            input_array: float array of object properties (eg fluxes in survey
                bands)
            k: int number of nearest neighbors to return.
        Returns:
            array of integer array indices of objects
        """
        
        tmp_array = (input_array - self._mean_array) / self._std_array
        d, i = self._kd_tree.query(tmp_array, k)
        return i, d[int(k/2)]
    
    def k_nearest_ball_point(self, input_array, max_dist):
        
        """
        Method to return the KDTree indicies from all points within a fixed
        distance of the point requested. The distance is expressed in sigma of
        the stored data array, i.e. a value of 1 returns all points within 1
        sigma.
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
        
        """
        Internal method for intilizing the KDTree object
        Args:
            self
        Returns:
            None
        """
        
        self._kd_tree = cKDTree(self._internal_array)
        
        return None
    
    def _normalize_data(self):
        
        """
        Internal method for scaling the data columns stored to a standard normal
        distribution of mean zero and standard deviation of 1.
        Args:
            self
        Returns:
            None
        """
        
        self._mean_array = self._internal_array.mean(axis = 0)
        self._std_array = self._internal_array.std(axis = 0)
        
        for col_idx in xrange(self._internal_array.shape[1]):
            self._internal_array[:, col_idx] = (
                (self._internal_array[:, col_idx] - self._mean_array[col_idx]) /
                self._std_array[col_idx])
        
        return None
    
    def get_mean_array(self):
        return self._mean_array
    
    def get_std_array(self):
        return self._std_array