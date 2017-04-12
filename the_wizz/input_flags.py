
"""The module contains all of the functions for loading flags and options from
the command line.
"""

from __future__ import division, print_function, absolute_import

import argparse


def parse_input_pdf_args():
    """Command line argument parser for the-wizz PDF creator. If you have a
    given survey file pair HDF5 file from the-wizz, you can set these arguments
    for the PDF maker portion of the library and get a robust estimate of the
    redshift distribution for your specific subsample of the survey catalog.
    ----------------------------------------------------------------------------
    Args:
        None
    Returns:
        ArgumentParser.parse_args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pair_hdf5_file', required=True,
                        type=str, help='Name of input HDF5 file to read '
                        'the pair counts between the spectroscopic and '
                        'unknown photometric data from.')
    parser.add_argument('--pair_scale_name', default='kpc30t300',
                        type=str, help='Name of the pair data scale to '
                        'load. This should be the name of the HDF5 group '
                        'the pair data is stored in. The format is '
                        'kpc[min]t[max].')
    parser.add_argument('--unknown_sample_file', required=True,
                        type=str, help='Name of unknown redshift '
                        'Photometric fits catalog containing the indices to '
                        'mask in the pair data file.')
    parser.add_argument('--unknown_index_name', required=True,
                        type=str, help='Name of unique object index for '
                        'the unknown objects. Indexes must be of type uint32')
    parser.add_argument('--unknown_weight_name', default=None,
                        type=str, help='Name of object weight for '
                        'the unknown objects.')
    parser.add_argument('--unknown_stomp_region_name', default=None,
                        type=str, help='Name of the column where the '
                        'STOMP region that each object belongs to is stored. '
                        'Setting this variable causes the code calculate the '
                        'over-densities relative to the average in the region '
                        'rather than globally. Useful if you are combining '
                        'several photometric, non-overlapying '
                        'surveys/pointings with different sensitivities.')
    parser.add_argument('--output_pdf_file_name', required=True,
                        type=str, help='Name of the output file to write '
                        'the resultant PDF to.')
    parser.add_argument('--z_min', default=0.01,
                        type=float, help='Minimum redshift for both the '
                        'pair_maker and pdf_maker.')
    parser.add_argument('--z_max', default=10.0,
                        type=float, help='Maximum redshift for both the '
                        'pair_maker and pdf_maker.')
    parser.add_argument('--z_n_bins', default=100,
                        type=int, help='Number of redshifts to specify '
                        'between z_min and z_max')
    parser.add_argument('--z_binning_type', default='linear',
                        type=str, help='Specify which type of binning to '
                        'use for the redshift bins. Choices are: '
                        '    linear: linear binning in redshift'
                        '    adapt: chose bins so that each has equal number '
                        'tarets.'
                        '    comoving: linear binning in comoving distance'
                        '    logspace: linear binning in ln(1 + z)')
    parser.add_argument('--use_inverse_weighting', action='store_true',
                        help='Use the inverse distance weighted columns from '
                        'the pair file instead of just a straight sum of '
                        'pairs.')
    parser.add_argument('--use_reference_cleaning', action='store_true',
                        help='Use a reference "cleaning" method similar to '
                        'that of Rahman et al. 2015. This downweights areas '
                        'of dense reference objects to produce a less noisy '
                        'clustering-z.')
    parser.add_argument('--n_bootstrap', default=1000, type=int,
                        help='Argument specifying the number of bootstrap '
                        'resamplings of the recovery to compute errors.')
    parser.add_argument('--n_processes', default=2, type=int,
                        help='Number of process to run. When computing large '
                        'angles it is recommended that several cores be '
                        'used(~4).')
    parser.add_argument('--n_reference_load_size', default=10000, type=int,
                        help='Number of reference pairs to load from the hdf5 '
                        'file at once. The chunk size should be set such that '
                        'the code has time to load the new data while it is '
                        'processing the current set.')
    parser.add_argument('--bootstrap_samples', default=None, type=str,
                        help='This is an optional argument specifying an '
                        'ascii file containing specified bootstrap samplings '
                        'to run. These should row-wise specifications of '
                        'regions from the input pair hdf5 file. Overrides '
                        'the number set in n_bootstrap.')
    parser.add_argument('--output_bootstraps_file', default=None, type=str,
                        help='This is an optional argument specifying an '
                        'ascii file to write the individual bootstrap pdfs '
                        'to.')
    parser.add_argument('--output_region_pickle_file', default=None,
                        type=str, help='This is an optional argument '
                        'specifying an output file to write a pickle of the '
                        'densities in each region. This can be used later to '
                        'combine with other surveys / pointings to create a '
                        'combined recovery. To see how the data is '
                        'stored/used look to the methods '
                        'write_region_densities and compute_pdf_bootstrap '
                        'methods in _pdf_maker_utils.py.')
    return _verify_none_type(parser.parse_args())


def parse_input_pdf_single_galaxy_args():
    """Command line argument parser for the-wizz PDF creator. If you have a
    given survey file pair HDF5 file from the-wizz, you can set these arguments
    for the PDF maker portion of the library and get a robust estimate of the
    redshift distribution for your specific subsample of the survey catalog.
    ----------------------------------------------------------------------------
    Args:
        None
    Returns:
        ArgumentParser.parse_args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pair_hdf5_file', required=True,
                        type=str, help='Name of input HDF5 file to read '
                        'the pair counts between the spectroscopic and '
                        'unknown photometric data from.')
    parser.add_argument('--pair_scale_name', default='kpc30t300',
                        type=str, help='Name of the pair data scale to '
                        'load. This should be the name of the HDF5 group '
                        'the pair data is stored in. The format is '
                        'kpc[min]t[max].')
    parser.add_argument('--unknown_sample_file', required=True,
                        type=str, help='Name of unknown redshift, '
                        'photometric fits catalog containing the indices to '
                        'match to the reference data and columns to match in '
                        'the kdtree. This should be the fits file created '
                        'that covers the reference redshift area.')
    parser.add_argument('--unknown_index_name', required=True,
                        type=str, help='Name of unique object index for '
                        'the unknown objects. Indexes must be of type uint32')
    parser.add_argument('--unknown_weight_name', default=None,
                        type=str, help='Name of object weight for '
                        'the unknown objects.')
    parser.add_argument('--unknown_stomp_region_name', default=None,
                        type=str, help='Name of the column where the '
                        'STOMP region that each object belongs to is stored. '
                        'Setting this variable causes the code calculate the '
                        'over-densities relative to the average in the region '
                        'rather than globally. Useful if you are combining '
                        'several photometric, non-overlapying '
                        'surveys/pointings with different sensitivities. This '
                        'catalog is read as is. Any selections or removal of '
                        'flagged values (eg -99, 99) must be beforehand or '
                        'they will corrupt the results.')
    parser.add_argument('--unknown_magnitude_names', required=True,
                        type=str, help='Comma separated list of fits '
                        'columns specifying the magnitudes to use in the '
                        'kdtree. This catalog is read as is. Any selections '
                        'or removal of flagged values (eg -99, 99) must be '
                        'beforehand or they will corrupt the results.')
    parser.add_argument('--use_as_colors', action='store_true',
                        help='Instead of using the raw '
                        'magnitudes, one can also use their information as a '
                        'difference or color. The columns will be taken as a '
                        'different in the order specifed. i.e. 0-1, 1-2, etc.')
    parser.add_argument('--unknown_other_names', default=None,
                        type=str, help='For any other catalog variable one '
                        'would like to use (eg type, size, fixed apature '
                        'flux) can be specified here in addition to the '
                        '"magnitudes" above. It is recommended to have no '
                        'more than around 10 columns in total. This catalog '
                        'is read as is. Any selections or removal of flagged '
                        'values (eg -99, 99) must be beforehand or they '
                        'will corrupt the results.')
    parser.add_argument('--match_sample_file', required=True,
                        type=str, help='Name of the fits file you would '
                        'to know the redshift distribution of each object. '
                        'It should have the same columns as the unknown '
                        'sample but need not cover the same area of the '
                        'reference objects. Each galaxy in this file will '
                        'be matched to a sample of galaxies in the unknown '
                        'sample usuing a kdtree. The return redshift '
                        'distribution is then the distribution for objects '
                        'in the unknown sample with similar properties to '
                        'the input object matched.')
    parser.add_argument('--n_kdtree_matched', default=1024,
                        type=int, help='The number of nearest neighbor '
                        'objects to match from the unknown sample kdtree to '
                        'the requested match objects. It is recommended to '
                        'attempt to have a least on average 100 unknown '
                        'sample objects per region.')
    parser.add_argument('--output_pdf_hdf5_file', required=True,
                        type=str, help='Name of the output hdf5_ file to '
                        'write the resultant PDFs to. The file structure is '
                        'similar to the output of pair_maker in that a PDF is '
                        'is stored as a HDF5 group named after the index of '
                        'each object and they are stored in a group named for '
                        'the scale run.')
    parser.add_argument('--z_min', default=0.01,
                        type=float, help='Minimum redshift for both the '
                        'pair_maker and pdf_maker.')
    parser.add_argument('--z_max', default=10.0,
                        type=float, help='Maximum redshift for both the '
                        'pair_maker and pdf_maker.')
    parser.add_argument('--z_n_bins', default=100,
                        type=int, help='Number of redshifts to specify '
                        'between z_min and z_max')
    parser.add_argument('--z_binning_type', default='linear',
                        type=str, help='Specify which type of binning to '
                        'use for the redshift bins. Choices are: '
                        '    linear: linear binning in redshift'
                        '    adaptive: chose bins so that each has equal '
                        'number tarets.'
                        '    comoving: bins equal in comving distance '
                        '    logspace: bins equal in ln(1 + z)')
    parser.add_argument('--use_inverse_weighting', action='store_true',
                        help='Use the inverse distance weighted columns from '
                        'the pair file instead of just a straight sum of '
                        'pairs.')
    parser.add_argument('--n_bootstrap', default=1000, type=int,
                        help='Argument specifying the number of bootstrap '
                        'resamplings of the recovery to compute errors.')
    parser.add_argument('--bootstrap_samples', default=None, type=str,
                        help='This is an optional argument specifying an '
                        'ascii file containing specified bootstrap samplings '
                        'to run. These should row-wise specifications of '
                        'regions from the input pair hdf5 file. Overrides '
                        'the number set in n_bootstrap.')
    parser.add_argument('--save_bootstraps', action='store_true',
                        help='Write the results of the individual bootstraps '
                        'to disk. This is only recommended to use when also '
                        'using the bootstrap_samples flag. Fair warning that '
                        'this will increase the ammount of storeage required '
                        'by roughly a factor of the number of bootstraps '
                        'requested.')
    parser.add_argument('--n_processes', default=1, type=int,
                        help='Number of process to run. When computing large '
                        'angles it is recommended that several cores be '
                        'used(~4).')
    parser.add_argument('--n_reference_load_size', default=10000, type=int,
                        help='Number of reference pairs to load from the hdf5 '
                        'file at once. The chunk size should be set such that '
                        'the code has time to load the new data while it is '
                        'processing the current set.')
    return _verify_none_type(parser.parse_args())


def parse_input_pair_args():
    """Command line argument parser for the-wizz pair finder portion of the
    library. If you already know how to use the STOMP library this should be
    relatively straight forward, easy to use, and safe for you. If you do not
    know anything about STOMP you will ikely be using the pdf creater methods
    and arguments.
    ----------------------------------------------------------------------------
    Args:
        None
    Returns:
        ArgumentParser.parse_args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--stomp_map', required=True,
                        type=str, help='Name of the STOMP map defining the '
                        'geometry on the sky for the reference and unknown '
                        'samples.')
    parser.add_argument('--n_regions', default=16,
                        type=int, help='Number of sub resgions to break up '
                        'the stomp map into for bootstrap/jackknifing. It is '
                        'recommended that the region size be no smaller than '
                        'the max scale requested in degrees.')
    parser.add_argument('--reference_sample_file', required=True,
                        type=str, help='Name of spectroscopic redshift '
                        'fits catalog.')
    parser.add_argument('--reference_ra_name', default='ALPHA_J2000',
                        type=str, help='Name of ra column in spectroscopic '
                        'fits file')
    parser.add_argument('--reference_dec_name', default='DELTA_J2000',
                        type=str, help='Name of dec column in '
                        'spectroscopic fits file')
    parser.add_argument('--reference_redshift_name', default='z_spec',
                        type=str, help='Name of redshift column in '
                        'spectroscopic fits file')
    parser.add_argument('--reference_index_name', default=None,
                        type=str, help='Name of unique object index for '
                        'the reference objects. Indexes must be of type '
                        'uint32')
    parser.add_argument('--unknown_sample_file', required=True,
                        type=str, help='Name of unknown redshift '
                        'Photometric fits catalog.')
    parser.add_argument('--unknown_ra_name', default='ALPHA_J2000',
                        type=str, help='Name of ra column in unknown, '
                        'photometric fits file')
    parser.add_argument('--unknown_dec_name', default='DELTA_J2000',
                        type=str, help='Name of dec column in '
                        'unknown, photometric fits file')
    parser.add_argument('--unknown_index_name', default=None,
                        type=str, help='Name of unique object index for '
                        'the unknown objects. Indexes must be of type uint32')
    parser.add_argument('--z_min', default=0.01,
                        type=float, help='Minimum redshift for both the '
                        'pair_maker and pdf_maker.')
    parser.add_argument('--z_max', default=10.0,
                        type=float, help='Maximum redshift for both the '
                        'pair_maker and pdf_maker.')
    parser.add_argument('--min_scale', default='30',
                        type=str, help='Comma seperated list of minimum '
                        'physical scales to measure the redshift recovery. '
                        'Expected units are physical kpc. Number of min '
                        'scales used should be equal to number of max scales '
                        'requested.')
    parser.add_argument('--max_scale', default='300',
                        type=str, help='Comma seperated list of maximum '
                        'physical scales to measure the redshift recovery. '
                        'Expected units are physical kpc. Number of max '
                        'scales used should be equal to number of min scales '
                        'requested.')
    parser.add_argument('--n_randoms', default=1,
                        type=int, help='Number of random iterations to '
                        'run for the natural estimator of 2-point '
                        'correlations. Number of uniform random points will '
                        'be n_randoms * # unknown objects.')
    parser.add_argument('--output_pair_hdf5_file', required=True,
                        type=str, help='Name of output HDF5 file to write '
                        'the pair counts between the spectroscopic and '
                        'unknown photometric data to.')
    return _verify_none_type(parser.parse_args())


def print_args(args):
    """Convenience function for printing the current value of the input
    arguments to the command line.
    ----------------------------------------------------------------------------
    Args:
        args: argparse object returned by ArgumentParser.parse_args()
    Returns:
        None
    """
    print("Current input flags...")
    for arg in vars(args):
        print("\t%s : %s" % (arg, getattr(args, arg)))
    return None


def _verify_none_type(args):
    """Function for safely handling if a user specifies "None" from the command
    line.
    ----------------------------------------------------------------------------
    Args:
        args: argparse object returned by ArgumentParser.parse_args()
    Returns:
        argparse object returned by ArgumentParser.parse_args()
    """
    for arg in vars(args):
        if getattr(args, arg) == 'None' or getattr(args, arg) == 'none' or \
           getattr(args, arg) == 'NONE':
            # Set argument value to None.
            setattr(args, arg, None)
    return args
