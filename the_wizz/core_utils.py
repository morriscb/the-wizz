
"""Utilities for The-wiZZ library. Contains file loading/closing, cosmology,
and setting the verbosity of the outputs.
"""

from __future__ import division, print_function, absolute_import

from astropy.cosmology import WMAP5
from astropy.io import fits
import h5py
import numpy as np


def file_checker_loader(input_file_name):
    """Utility function for checking the existence of a file and loading the
    file with the proper format. Currently checks for FITS files.
    ----------------------------------------------------------------------------
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
    if input_file_name.endswith('fit') or input_file_name.endswith('fits') or \
       input_file_name.endswith('gz') or input_file_name.endswith('cat'):
        hdu_list = fits.open(input_file_name)
        data = hdu_list[1].data
        return data
    elif input_file_name.endswith('hdf5') or input_file_name.endswith('dat'):
        hdf5_file = h5py.File(input_file_name, 'r')
        return hdf5_file
    else:
        print("File type not currently supported. Try again later. "
              "The-wiZZ is exiting.")
        raise IOError
    return None


def create_hdf5_file(hdf5_file_name, args):
    # TODO:
    #     Decide if I want to use libver latest or not. Could be more stable
    #     if we use the "earliest" version. Will have to speed test saving
    #     and loading of the pairs.
    """Convenience function for creating an HDF5 file with attributes set in
    input_flags. Saves the current input flags to the group input_flags for
    later reference
    ----------------------------------------------------------------------------
    Args:
        hdf5_file_name: string name of the HDF5 file to create
        args: argparse ArgumentParser.parse_args object from input_flags
    Returns:
        open HDF5 file object
    """
    hdf5_file = h5py.File(hdf5_file_name, 'w-', libver='latest')
    if args is not None:
        flag_grp = hdf5_file.create_group('input_flags')
        for arg in vars(args):
            if getattr(args, arg) is None:
                flag_grp.attrs.create(arg, 'None')
            else:
                flag_grp.attrs.create(arg, getattr(args, arg))
    return hdf5_file


def create_ascii_file(ascii_file_name, args):
    """Convenience function for creating an output ascii file. This method
    writes the current state of the input_flags arguments to the header of the
    file and returns an open Python file handle object. The method will over
    write any file it is given so use with caution.
    ----------------------------------------------------------------------------
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
