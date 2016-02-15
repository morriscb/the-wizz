#! /usr/bin/python

import __core__
import _input_flags
import _pdf_maker_utils
import h5py
import numpy as np
import sys

if __name__ == "__main__": 
    
    ### First we parse the command line for arguments as usual. See
    ### _input_flags.py for a full list of input arguments. 
    args = _input_flags.parse_input_args()
    
    ### Load the file containing all matched pairs of spectroscopic and
    ### photometric objects.
    hdf5_pair_file = __core__.load_pair_hdf5(args.input_pair_hdf5_file)
    
    ### Now we figure out what kind of redshift binning we would like to have.
    ### This will be one of the largest impacts on the signal to noise of the
    ### measurement. Some rules of thumb are:
    ###     The narrower bins are in redshift the better. You are measuring a
    ### correlation, the narrower the bin size in comoving distance the more
    ### correlated things will be and thus increase the amplitude. Aka use
    ### Groth/Pebbles[sic] scaling to your advantage.
    ###     For a spectroscopic sample that is selected for a specific redshift
    ### range with few galaxies outside that range (eg DEEP2), adaptive binning
    ### is recommended. This will keep a equal number spectra per redshift bin.
    ### A good rule is to try to have about 100 spectra per redshift bin for max
    ### signal to noise.
    ###     Linear binning is provided as a curtesy and is not nesassarly
    ### recommended. It will not give the best signal to noise compared to
    ### adaptive and has the same draw backs as adaptive is that the bias could
    ### be changing oddly from bin to bin. It is recommended that the user try
    ### adaptive and comoving spaced bins for the best results. Comoving returns
    ### bins that are of equal comoving distance from the line of sight.
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
    
    ### This is where the heavy lifting happens. We create our PDF maker object
    ### which will hold the pair file for use, calculate the over density per
    ### redshift bin, and also store intermediary results for later use. 
    pdf_maker = _pdf_maker_utils.PDFMaker(hdf5_pair_file)
    ### This is a test method for writting out the pdf for a fixed redshift bin.
    ### This will be replaced by a function using an array of requested id's as
    ### input, allowing for recovery of any subsample from the data.
    pdf_maker.straight_sum('kpc30t300', z_bin_edge_array, args.z_max)
    ### Simple function to write out the resultant over-density with redshift or
    ### PDF.
    pdf_maker.write_to_ascii('test_z0.5t0.7_kpc30t300.ascii')
    
    ### TODO:
    ###     PDFMaker: Need to impliment and call the method to exclude with a
    ### list of galaxy indices.
    ###     PDFMaker: make the methods for writing out results more descirptive.
    