#! /usr/bin/python

"""Program to create single galaxy clustering redshifts from creating a kdTree
on on properties an input catalog of objects and a set of objects to match into
said kdTree. After the objects are matched in the code runs pdf_maker as normal
and outputs the results for eatch object a match was requested of into a HDF5
file.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from the_wizz import core_utils
from the_wizz import kdtree_utils
from the_wizz import pdf_maker_utils
from the_wizz import input_flags

if __name__ == "__main__":
    print("")
    print("The-wiZZ has begun conjuring: running pair maker for single "
          "galaxies...")
    # First we parse the command line for arguments as usual. See
    # input_flags.py for a full list of input arguments.
    args = input_flags.parse_input_pdf_single_galaxy_args()
    input_flags.print_args(args)
    # Create the HDF5 file we will write out to store the PDFs.
    output_pdf_hdf5_file = core_utils.create_hdf5_file(
        args.output_pdf_hdf5_file, args)
    # Load the file containing all matched pairs of spectroscopic and
    # photometric objects.
    print("Loading files...")
    hdf5_pair_file = core_utils.file_checker_loader(args.input_pair_hdf5_file)
    unknown_data = core_utils.file_checker_loader(args.unknown_sample_file)
    match_data = core_utils.file_checker_loader(args.match_sample_file)
    # Load the spectroscopic data from the HDF5 data file.
    print("Preloading reference data...")
    pdf_maker = pdf_maker_utils.PDFMaker(hdf5_pair_file[args.pair_scale_name],
                                          args)
    reference_pair_data = pdf_maker_utils._load_pair_data(
        hdf5_pair_file[args.pair_scale_name], 0,
        pdf_maker.reference_redshift_array.shape[0])
    if pdf_maker.reference_redshift_array.max() < args.z_max:
        print("WARNING: requested z_max is greater than available reference "
              "redshifts.")
    # Now we figure out what kind of redshift binning we would like to have.
    # This will be one of the largest impacts on the signal to noise of the
    # measurement. Some rules of thumb are:
    #     The narrower bins are in redshift the better. You are measuring a
    # correlation, the narrower the bin size in comoving distance the more
    # correlated things will be and thus increase the amplitude. Aka use
    # Groth/Pebbles[sic] scaling to your advantage.
    #     For a spectroscopic sample that is selected for a specific redshift
    # range with few galaxies outside that range (eg DEEP2), adaptive binning
    # is recommended. This will keep a equal number spectra per redshift bin.
    # A good rule is to try to have about 100 spectra per redshift bin for max
    # signal to noise.
    #     Linear binning is provided as a curtesy and is not nesassarly
    # recommended. It will not give the best signal to noise compared to
    # adaptive and has the same draw backs as adaptive is that the bias could
    # be changing oddly from bin to bin. It is recommended that the user try
    # adaptive and comoving spaced bins for the best results. Comoving returns
    # bins that are of equal comoving distance from the line of sight.
    print("Creating bins...")
    if args.z_binning_type == 'linear':
        z_bin_edge_array = pdf_maker_utils._create_linear_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins)
    elif args.z_binning_type == 'adaptive':
        z_bin_edge_array = pdf_maker_utils._create_adaptive_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins,
            pdf_maker.reference_redshift_array)
    elif args.z_binning_type == 'comoving':
        z_bin_edge_array = pdf_maker_utils._create_comoving_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins)
    elif args.z_binning_type == 'logspace':
        z_bin_edge_array = pdf_maker_utils._create_logspace_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins)
    else:
        print("Requested binning name invalid. Valid types are:")
        print("\tlinear: linear binning in redshift")
        print("\tadaptive: constant reference objects per redshift bin")
        print("\tcomoving: linear binning in comoving distance")
        print("Retunning linear binning...")
        z_bin_edge_array = pdf_maker_utils._create_linear_redshift_bin_edges(
            args.z_min, args.z_max, args.z_n_bins)
    # Now that we know the redshift array, we create the group where we will
    # store the output pdfs.
    scale_group = output_pdf_hdf5_file.create_group(args.pair_scale_name)
    scale_group.attrs.create('redshift_lower_edge_array', z_bin_edge_array)
    # This portion of the code creates the kdtree on the unknown sample that
    # we will match our match sample to.
    print("Making KDTree")
    mag_name_list = args.unknown_magnitude_names.split(',')
    try:
        other_name_list = args.unknown_other_names.split(',')
    except AttributeError:
        other_name_list = []
    print(mag_name_list, other_name_list)
    unknown_value_array = kdtree_utils.create_match_data(
        unknown_data, mag_name_list, other_name_list, args.use_as_colors)
    print(unknown_value_array.shape, unknown_data.shape)
    kdtree = kdtree_utils.CatalogKDTree(unknown_value_array)
    # Now we create the same array as the unknown sample for our match
    # sample.
    print("Creating matching data array...")
    match_data_array = kdtree_utils.create_match_data(
        match_data, mag_name_list, other_name_list, args.use_as_colors)
    # Now we loop over each match objects data array, match it to the unknown
    # sample's kdtree and submit those galaxies to pdf_maker to create an
    # estimate of the pdf from the unknown sample objects with the closest
    # properites.
    print("Starting match object loop...")
    for match_idx, match_obj in enumerate(match_data_array):
        print("\tCurrent idx: %i" % match_idx)
        pdf_maker.reset_pairs()
        # Match the ids.
        id_array, quartile_dists = kdtree(match_obj, args.n_kdtree_matched)
        kdtree_utils.collapse_ids_to_single_estimate(
            hdf5_pair_file[args.pair_scale_name], reference_pair_data, pdf_maker,
            unknown_data[id_array], args)
        # Get the region densities.
        pdf_maker.compute_region_densities(z_bin_edge_array, args.z_max)
        # Collapse the region densities and estimate the error.
        if args.bootstrap_samples is None:
            pdf_maker.compute_pdf_bootstrap(args.n_bootstrap)
        else:
            bootstrap_region_array = np.loadtxt(args.bootstrap_samples,
                                                dtype=np.int_)
            pdf_maker._compute_pdf_bootstrap(bootstrap_region_array)
        tmp_grp = scale_group.create_group(
            '%i' % match_data[args.unknown_index_name][match_idx])
        tmp_grp.attrs.create('one_quarter_dist', quartile_dists[0])
        tmp_grp.attrs.create('median_dist', quartile_dists[1])
        tmp_grp.attrs.create('three_quarter_dist', quartile_dists[2])
        tmp_grp.attrs.create('max_dist', quartile_dists[3])
        tmp_grp.create_dataset('pdf', data=np.array(pdf_maker.density_array,
                                                    dtype=np.float32),
                               compression='lzf', shuffle=True)
        tmp_grp.create_dataset('pdf_err',
                               data=np.array(pdf_maker.density_err_array,
                                             dtype=np.float32),
                               compression='lzf', shuffle=True)
        if args.save_bootstraps:
            tmp_grp.create_dataset('bootstraps',
                                   data=np.array(pdf_maker.bootstrap_array,
                                                 dtype=np.float32),
                                   compression='lzf', shuffle=True)
    # Now we close out the hdf5 file.
    output_pdf_hdf5_file.close()
    print("Done!")
