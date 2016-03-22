import argparse


def parse_input_pdf_args():
    
    """
    Command line argument parser for The-wiZZ PDF creator. If you have a given
    survey file pair HDF5 file from The-wiZZ, you can set these arguments for
    the PDF maker portion of the library and get a robust estimate of the 
    redshift distribution for your specific subsample of the survey catalog.
    Args:
        None
    Returns:
        ArgumentParser.parse_args object
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_pair_hdf5_file', required = True,
                        type = str, help = 'Name of input HDF5 file to read '
                        'the pair counts between the spectroscopic and unknown '
                        'photometric data from.')
    parser.add_argument('--pair_scale_name', default = 'kpc30t300',
                        type = str, help = 'Name of the pair data scale to '
                        'load. This should be the name of the HDF5 group '
                        'the pair data is stored in. The format is '
                        'kpc[min]t[max].')
    parser.add_argument('--unknown_sample_file', required = True,
                        type = str, help = 'Name of unknown redshift '
                        'Photometric fits catalog containing the indices to '
                        'mask in the pair data file.')
    parser.add_argument('--unknown_index_name', default = 'SeqNr',
                        type = str, help = 'Name of unique object index for '
                        'the unknown objects. Indexes must be of type uint32')
    parser.add_argument('--unknown_weight_name', default = None,
                        type = str, help = 'Name of object weight for '
                        'the unknown objects.')
    parser.add_argument('--output_pdf_file_name', required = True,
                        type = str, help = 'Name of the output file to write '
                        'the resultant PDF to.')
    parser.add_argument('--z_min', default = 0.01,
                        type = float, help = 'Minimum redshift for both the '
                        'pair_maker and pdf_maker.')
    parser.add_argument('--z_max', default = 10.0,
                        type = float, help = 'Maximum redshift for both the '
                        'pair_maker and pdf_maker.')
    parser.add_argument('--z_n_bins', default = 100,
                        type = float, help = 'Number of redshifts to specify '
                        'between z_min and z_max')
    parser.add_argument('--z_binning_type', default = 'linear',
                        type = str, help = 'Specify which type of binning to '
                        'use for the redshift bins. Choices are: '
                        '    linear: linear binning in redshift'
                        '    adapt: chose bins so that each has equal number '
                        'tarets.'
                        '    comoving: linear binning in comoving distance')
    parser.add_argument('--use_inverse_weighting', action = 'store_true',
                        help = 'Use the inverse distance weighted columns from '
                        'the pair file instead of just a straight sum of '
                        'pairs.')
    parser.add_argument('--n_bootstrap', default = 1000, type = int,
                        help = 'Argument specifying the number of bootstrap '
                        'resamplings of the recovery to compute errors.')
    parser.add_argument('--bootstrap_samples', default = None, type = str,
                        help = 'This is an optional argument specifying an '
                        'ascii file containing specified bootstrap samplings '
                        'to run. These should row-wise specifications of '
                        'regions from the input pair hdf5 file. Overrides '
                        'the number set in n_bootstrap.')
    
    
    return parser.parse_args()

def parse_input_pair_args():
    
    """
    Command line argument parser for The-wiZZ pair finder portion of the
    library. If you already know how to use the STOMP library this should be 
    relatively straight forward, easy to use, and safe for you. If you do not
    know anything about STOMP you will ikely be using the pdf creater methods
    and arguments.
    Args:
        None
    Returns:
        ArgumentParser.parse_args object
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--stomp_map', required = True,
                        type = str, help = 'Name of the STOMP map defining the '
                        'geometry on the sky for the target and unknown '
                        'samples.')
    parser.add_argument('--n_regions', default = 16,
                        type = int, help = 'Number of sub resgions to break up '
                        'the stomp map into for bootstrap/jackknifing. It is '
                        'recommended that the region size be no smaller than '
                        'the max scale requested in degrees.')
    parser.add_argument('--target_sample_file', required = True,
                        type = str, help = 'Name of spectroscopic redshift '
                        'fits catalog.')
    parser.add_argument('--target_ra_name', default = 'ALPHA_J2000',
                        type = str, help = 'Name of ra column in spectroscopic '
                        'fits file')
    parser.add_argument('--target_dec_name', default = 'DELTA_J2000',
                        type = str, help = 'Name of dec column in '
                        'spectroscopic fits file')
    parser.add_argument('--target_redshift_name', default = 'z_spec',
                        type = str, help = 'Name of redshift column in '
                        'spectroscopic fits file')
    parser.add_argument('--target_index_name', default = None,
                        type = str, help = 'Name of unique object index for '
                        'the target objects. Indexes must be of type uint32')
    parser.add_argument('--unknown_sample_file', required = True,
                        type = str, help = 'Name of unknown redshift '
                        'Photometric fits catalog.')
    parser.add_argument('--unknown_ra_name', default = 'ALPHA_J2000',
                        type = str, help = 'Name of ra column in unknown, '
                        'photometric fits file')
    parser.add_argument('--unknown_dec_name', default = 'DELTA_J2000',
                        type = str, help = 'Name of dec column in '
                        'unknown, photometric fits file')
    parser.add_argument('--unknown_index_name', default = None,
                        type = str, help = 'Name of unique object index for '
                        'the unknown objects. Indexes must be of type uint32')
    parser.add_argument('--z_min', default = 0.01,
                        type = float, help = 'Minimum redshift for both the '
                        'pair_maker and pdf_maker.')
    parser.add_argument('--z_max', default = 10.0,
                        type = float, help = 'Maximum redshift for both the '
                        'pair_maker and pdf_maker.')
    parser.add_argument('--min_scale', default = '30',
                        type = str, help = 'Coma seperated list of minimum '
                        'physical scales to measure the redshift recovery. '
                        'Expected units are physical kpc. Number of min '
                        'scales used should be equal to number of max scales '
                        'requested.')
    parser.add_argument('--max_scale', default = '300',
                        type = str, help = 'Coma seperated list of maximum '
                        'physical scales to measure the redshift recovery. '
                        'Expected units are physical kpc. Number of max '
                        'scales used should be equal to number of min scales '
                        'requested.')
    parser.add_argument('--n_randoms', default = 1,
                        type = int, help = 'Number of random iterations to '
                        'run for the natural estimator of 2-point '
                        'correlations. Number of uniform random points will be '
                        'n_randoms * # unknown objects.')
    parser.add_argument('--output_pair_hdf5_file', required = True,
                        type = str, help = 'Name of output HDF5 file to write '
                        'the pair counts between the spectroscopic and unknown '
                        'photometric data to.')
    
    return parser.parse_args()

def print_args(args):
    """
    Convenience function for printing the current value of the input arguments
    to the command line.
    Args:
        args: argparse object returned by ArgumentParser.parse_args()
    Returns:
        None
    """
    
    print("Current input flags...")
    for arg in vars(args):
        print("\t%s : %s" % (arg, getattr(args, arg)))
    
    return None

