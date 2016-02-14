import argparse

"""
The Redshift wiZZard.
Adapted from Mubdi Rahman's wizard engine. Rahman et al. 2015

_input_flags.py

Definitions for command line parsing for The Redshift wiZZard.
"""

def parse_input_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stomp_map', default = '',
                        type = str, help = 'Name of the STOMP map defining the '
                        'geometry on the sky for the target and unknown '
                        'samples.')
    parser.add_argument('--n_regions', default = 16,
                        type = str, help = 'Number of sub resgions to break up '
                        'the stomp map into for bootstrap/jackknifing. It is '
                        'recommended that the region size be no smaller than '
                        'the max scale requested in degrees.')
    parser.add_argument('--target_sample_file', default = '',
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
    parser.add_argument('--unknown_sample_file', default = '',
                        type = str, help = 'Name of unknown redshift '
                        'photometric fits catalog.')
    parser.add_argument('--unknown_ra_name', default = 'ALPHA_J2000',
                        type = str, help = 'Name of ra column in unknown, '
                        'photometric fits file')
    parser.add_argument('--unknown_dec_name', default = 'DELTA_J2000',
                        type = str, help = 'Name of dec column in '
                        'unknown, photometric fits file')
    parser.add_argument('--unknown_index_name', default = None,
                        type = str, help = 'Name of unique object index for '
                        'the unknown objects. Indexes must be of type uint32')
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
    parser.add_argument('--output_pair_hdf5_file', default = '',
                        type = str, help = 'Name of output HDF5 file to write '
                        'the pair counts between the spectroscopic and unknown '
                        'photometric data to.')
    parser.add_argument('--input_pair_hdf5_file', default = '',
                        type = str, help = 'Name of input HDF5 file to read '
                        'the pair counts between the spectroscopic and unknown '
                        'photometric data from.')
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
    return parser.parse_args()