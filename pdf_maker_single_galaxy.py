#! /usr/bin/python

import _core_utils
import input_flags
import _kdtree_utils
import _pdf_maker_utils
import h5py
import numpy as np
import sys

if __name__ == "__main__": 
    
    
    ### First we parse the command line for arguments as usual. See
    ### input_flags.py for a full list of input arguments. 
    args = input_flags.parse_input_pdf_single_galaxy_args()
    
    print("")
    print("The-wiZZ has begun conjuring: running pair maker for single "
          "galaxies...")
    
    input_flags.print_args(args)
    
    ### Load the file containing all matched pairs of spectroscopic and
    ### photometric objects.
    print("Loading files...")
    hdf5_pair_file = _core_utils.file_checker_loader(args.input_pair_hdf5_file)
    unknown_data = _core_utils.file_checker_loader(args.unknown_sample_file)
    match_data = _core_utils.file_checker_loader(args.match_sample_file)
    ### Load the spectroscopic data from the HDF5 data file.
    print("Preloading target data...")
    pdf_maker = _pdf_maker_utils.PDFMaker(hdf5_pair_file[args.pair_scale_name],
                                          args)
    if pdf_maker.target_redshift_array.max() < args.z_max:
        print("WARNING: requested z_max is greater than available target "
              "redshifts.")
        args.z_max = pdf_maker.target_redshift_array.max() + 1e-16
        print("\tResetting to %.4f..."  % args.z_max)
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
    print("Creating bins...")
    if args.z_binning_type == 'linear':
        z_bin_edge_array = _pdf_maker_utils._create_linear_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins)
    elif args.z_binning_type == 'adaptive':
        z_bin_edge_array = _pdf_maker_utils._create_adaptive_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins,
            pdf_maker.target_redshift_array)
    elif args.z_binning_type == 'comoving':
        z_bin_edge_array = _pdf_maker_utils._create_comoving_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins)
    else:
        print("Requested binning name invalid. Valid types are:")
        print("\tlinear: linear binning in redshift")
        print("\tadaptive: constant target objects per redshift bin")
        print("\tcomoving: linear binning in comoving distance")
        print("Retunning linear binning...")
        z_bin_edge_array = _pdf_maker_utils._create_linear_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins)
    
    ### This portion of the code creates the kdtree on the unknown sample that 
    ### we will match our match sample to. 
    print("Making KDTree")
    mag_name_list = args.unknown_magnitude_names.split(',')
    try:
        other_name_list = args.unknown_other_names.split(',')
    except AttributeError:
        other_name_list = []
    print mag_name_list, other_name_list
    kdtree = _kdtree_utils.CatalogKDTree(
        _kdtree_utils.create_match_data(
            unknown_data, mag_name_list, other_name_list, args.use_as_colors))
    
    ### Now we create the same array as the unknown sample for our match
    ### sample.
    print("Creating matching data array...")
    match_data_array = _kdtree_utils.create_match_data(
        match_data, mag_name_list, other_name_list, args.use_as_colors)
    match_data_density_array = np.empty((match_data_array.shape[0],
                                        z_bin_edge_array.shape[0]))
    
    ### Now we loop over each match objects data array, match it to the unknown
    ### sample's kdtree and submit those galaxies to pdf_maker to create an
    ### estimate of the pdf from the unknown sample objects with the closest
    ### properites.
    print("Starting match object loop...")
    for match_idx, match_obj in enumerate(match_data_array):
        
        ### match the ids
        _pdf_maker_utils.collapse_ids_to_single_estimate(
            hdf5_pair_file[args.pair_scale_name], pdf_maker,
            unknown_data[kdtree(match_obj, args.n_kdtree_matched)], args)
        
        ### Get the region densities
        pdf_maker.compute_region_densities(z_bin_edge_array, args.z_max)
        
        ### Collapse the region densities and estimate the error.
        if args.bootstrap_samples is None:
            pdf_maker.compute_pdf_bootstrap(args.n_bootstrap)
        else:
            bootstrap_region_array = np.loadtxt(args.bootstrap_samples,
                                            dtype = np.int_)
            pdf_maker._compute_pdf_bootstrap(bootstrap_region_array)
        
        match_data_density_array[match_idx, :] = pdf_maker.density_array
        
    ### Now we create the output header for the output ascii file.
    ### TODO:
    ###     Maybe change this output to HDF5
    output_header = 'input_flags:\n'
    for arg in vars(args):
        output_header += '\t%s : %s\n' % (arg, getattr(args, arg))
    output_header += "redshifts : "
    for z in pdf_maker.redshift_array:
        output_header += '%.8e ' % z
    output_header += '\n'
    
    ### Write out the ascii file.
    np.savetxt(args.output_pdf_file_name, match_data_density_array, 
               fmt = '%.8e', header = output_header)
    
    print "Done!"