### TODO:
###     split core into two with one having the stomp calls and one not.

from astropy.cosmology import Planck13
from astropy.io import fits
import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iu_spline
import sys
from __builtin__ import None


"""
Utilities for The-wiZZ library. Contains file loading/closing, cosmology, and 
setting the verbosity of the outputs.
"""


verbose = False
_initialized_cosmology = False
_comov_dist_to_redshift_spline = None


def set_verbose(v_bool):
    
    """
    Set the internal verbosity of the library at the time of running.
    Args:
        v_bool: boolean variable specifying if the code should be verbose or not
    Returns:
        None
    """
    
    verbose = v_bool
    return None

def verbose_print_statement(statement_string):
    
    pass

def _initialize_cosmology():
    
    """
    Initlized internal __core__ variables storing the trend of redshift vs
    comoving distance. Default cosmology is from Planck13.
    Args:
        None
    Returns:
        None
    """
    
    redshift_array = np.linspace(0.0, 10.0, 1000)
    comov_array = Planck13.comoving_distance(redshift_array)
    _comov_dist_to_redshift_spline = iu_spline(comov_array, redshift_array)
    _initilized_cosmology = True

def redshift(comov_dist):
    
    """
    Spline wrapper for converting a comoving line of sight distance into a
    redshift assuming the Planck13 cosmology.
    Args:
        comov_dist: float or float array cosmoving distance in Mpc
    Returns:
        float or float array redshift
    """
    
    if not _initialized_cosmology:
        _initialize_cosmology()
    return _comov_dist_to_redshift_spline(comov_dist)

def file_checker_loader(sample_file_name):
    
    ### TODO:
    ###     Clean up this function and add more supported files types. Combine
    ###     with the hdf5 loads.
    
    """
    Utility function for checking the existence of a file and loading the file
    with the proper format. Currently checks for FITS files.
    Args:
        sample_file_name: name of file on disk to load
    Returns:
        open file object data
    """
    
    try:
        file_handle = open(sample_file_name)
        file_handle.close()
    except IOError:
        print("IOError: File %s not found. The-wiZZ is exiting." %
              sample_file_name)
        raise IOError("File not found.")
    
    data_type = sample_file_name.split('.')[-1]
    if data_type == 'fit' or data_type == 'fits' or data_type == 'cat':
        
        hdu_list = fits.open(sample_file_name)
        data = hdu_list[1].data
        hdu_list.close()
        return data
    else:
        print("File type not currently supported. Try again later. "
              "The-wiZZ is exiting.")
        raise IOError
    
    return None

def create_hdf5_file(hdf5_file_name, args):
    
    """
    Convenience function for creating an HDF5 file with attributes set in
    _input_flags.
    Args:
        hdf5_file_name: string name of the HDF5 file to create
        args: argparse ArgumentParser.parse_args object from _input_flags
    Returns:
        open HDF5 file object
    """
    
    hdf5_file = h5py.File(hdf5_file_name, 'w-', libver = 'latest')
    
    return hdf5_file

def load_pair_hdf5(hdf5_file_name):
    
    ### TODO:
    ###     Possibly move this into the file_checker_loader function
    
    """
    Convenience function for loading an HDF5 wiZZ pair file.
    Args:
        hdf5_file_name: string name of the wiZZ HDF5 pair file to load
    Returns:
        open HDF5 file object
    """
    
    hdf5_file = h5py.File(hdf5_file_name, 'r')
    
    return hdf5_file

def close_hdf5_file(hdf5_file):
    
    """
    Convenience function for closing an open HDF5 file object
    args:
        hdf5_file: hdf5 file object
    Returns:
        None
    """
    
    hdf5_file.close()
    
    return None
    