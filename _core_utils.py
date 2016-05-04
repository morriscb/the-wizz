### TODO:
###     split core into two with one having the stomp calls and one not.

from astropy.cosmology import WMAP5
from astropy.io import fits
import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iu_spline
import sys


"""
Utilities for The-wiZZ library. Contains file loading/closing, cosmology, and 
setting the verbosity of the outputs.
"""


global verbose
global _initialized_cosmology
 
verbose = False
_initialized_cosmology = False

def set_verbose(v_bool):
    
    """
    Set the internal verbosity of the library at the time of running.
    Args:
        v_bool: boolean variable specifying if the code should be verbose or not
    Returns:
        None
    """
    
    global verbose
    verbose = v_bool
    return None

def verbose_print_statement(statement_string):
    
    if verbose:
        print(statement_string)
    return None

def _initialize_cosmology():
    
    """
    Initlized internal __core__ variables storing the trend of redshift vs
    comoving distance. Default cosmology is from WMAP5.
    Args:
        None
    Returns:
        None
    """
    
    ### TODO:
    ###     Talk to someone about this. Are globals the best way to do this.
    redshift_array = np.linspace(0.0, 10.0, 1000)
    comov_array = WMAP5.comoving_distance(redshift_array)
    global _comov_dist_to_redshift_spline
    _comov_dist_to_redshift_spline = iu_spline(comov_array, redshift_array)
    global _initilized_cosmology
    _initilized_cosmology = True
    

def redshift(comov_dist):
    
    """
    Spline wrapper for converting a comoving line of sight distance into a
    redshift assuming the WMAP5 cosmology.
    Args:
        comov_dist: float or float array cosmoving distance in Mpc
    Returns:
        float or float array redshift
    """
    
    if not _initialized_cosmology:
        _initialize_cosmology()
    return _comov_dist_to_redshift_spline(comov_dist)

def file_checker_loader(input_file_name):
    
    """
    Utility function for checking the existence of a file and loading the file
    with the proper format. Currently checks for FITS files.
    Args:
        sample_file_name: name of file on disk to load
    Returns:
        open file object data
    """
    
    try:
        file_handle = open(input_file_name)
        file_handle.close()
    except IOError:
        print("IOError: File %s not found. The-wiZZ is exiting." %
              input_file_name)
        raise IOError("File not found.")
    
    data_type = input_file_name.split('.')[-1]
    
    if data_type == 'fit' or data_type == 'fits' or data_type == 'cat':
        
        hdu_list = fits.open(input_file_name)
        data = hdu_list[1].data
        return data
    
    if data_type == 'hdf5' or data_type == 'dat':
        
        hdf5_file = h5py.File(input_file_name, 'r')
        
        return hdf5_file
        
    else:
        print("File type not currently supported. Try again later. "
              "The-wiZZ is exiting.")
        raise IOError
    
    return None

def create_hdf5_file(hdf5_file_name, args):
    
    ### TODO:
    ###     Decide if I want to use libver latest or not. Could be more stable 
    ###     if we use the "earliest" version. Will have to speed test saving 
    ###     and loading of the pairs.
    
    """
    Convenience function for creating an HDF5 file with attributes set in
    input_flags. Saves the current input flags to the group input_flags for
    later reference
    Args:
        hdf5_file_name: string name of the HDF5 file to create
        args: argparse ArgumentParser.parse_args object from input_flags
    Returns:
        open HDF5 file object
    """
    
    hdf5_file = h5py.File(hdf5_file_name, 'w-', libver = 'latest')
    flag_grp = hdf5_file.create_group('input_flags')
    
    for arg in vars(args):
        if getattr(args, arg) is None:
            flag_grp.attrs.create(arg, 'None')
        else:
            flag_grp.attrs.create(arg, getattr(args, arg))
    
    return hdf5_file

def create_ascii_file(ascii_file_name, args):
    
    """
    Convenience function for creating an output ascii file. This method writes 
    the current state of the input_flags arguments to the header of the file and
    returns an open Python file handle object. The method will over write any
    file it is given so use with caution.
    Args:
        ascii_file_name: string name of the file to write too
        args: argparse ArgumentParser.parse_args object from input_flags
    Returns:
        open Python file object
    """
    
    ascii_file = open(ascii_file_name, 'w')
    
    ascii_file.writelines('# input_flags:\n')
    for arg in vars(args):
        ascii_file.writelines('#\t%s : %s\n' % (arg, getattr(args, arg)))
    
    return ascii_file 
    