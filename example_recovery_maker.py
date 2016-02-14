#! /usr/bin/python

import __core__
import _input_flags
import _pdf_maker_utils
import h5py
import numpy as np
import sys

if __name__ == "__main__":
    
    args = _input_flags.parse_input_args()
    
    hdf5_pair_file = __core__.load_pair_hdf5(args.input_pair_hdf5_file)
    
    if args.z_binning_type == 'linear':
        z_bin_edge_array = _pdf_maker_utils._create_linear_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins)
    elif args.z_binning_type == 'adaptive':
        
        z_bin_edge_array = _pdf_maker_utils._create_adaptive_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins, target_redshift_array)
    elif args.z_binning_type == 'comoving':
        z_bin_edge_array = _pdf_maker_utils._create_comoving_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins)
    else:
        print("Requested binning name invalid. Valid types are:")
        print("\tlinear: linear binning in redshift")
        print("\tadaptive: constant target objects per redshift bin")
        print("\tcomoving: linear binning in comoving distance")
        print("\nThe wiZZ is exitting.")
        sys.exit()
        
    print z_bin_edge_array
        
    pdf_maker = _pdf_maker_utils.PDFMaker(hdf5_pair_file)
    pdf_maker.straight_sum('kpc30t300', z_bin_edge_array, args.z_max)
    pdf_maker.write_to_ascii('test_z0.5t0.7_kpc30t300.ascii')
    