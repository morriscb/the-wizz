
import argparse
import numpy as np
import pickle

"""
This program allows for the combination of different the-wizz, pdf_maker runs
after the fact using the region pickle files output from the code. This can be
useful for combining different recoveries from different spectroscoic pointings
in a way that is internally consistent.
"""


def load_from_pickle(file_name_list):

    region_dict = {'n_regions': 0,
                   'redshift': np.array([]),
                   'n_reference': np.array([]),
                   'unknown': np.array([]),
                   'rand': np.array([]),
                   'area': np.array([]),
                   'resolution': np.array([])}

    for file_idx, file_name in enumerate(file_name_list):

        pkl_file = open(file_name)
        region_density_dict = pickle.load(pkl_file)
        pkl_file.close()

        region_dict['n_regions'] += region_density_dict['n_regions']
        if file_idx == 0:
            region_dict['redshift'] = region_density_dict['redshift']
            region_dict['n_reference'] = region_density_dict[
                'n_reference'].astype(dtype='float')
            region_dict['unknown'] = region_density_dict[
                'unknown'].astype(dtype='float')
            region_dict['rand'] = region_density_dict[
                'rand'].astype(dtype='float')
            region_dict['area'] = region_density_dict['area']
            region_dict['resolution'] = region_density_dict[
                'resolution'].astype(dtype='float')
        else:
            region_dict['redshift'] = np.concatenate(
                (region_dict['redshift'], region_density_dict['redshift']),
                axis=1)
            region_dict['n_reference'] = np.concatenate(
                (region_dict['n_reference'],
                 region_density_dict['n_reference']),
                axis=1)
            region_dict['unknown'] = np.concatenate(
                (region_dict['unknown'], region_density_dict['unknown']),
                axis=1)
            region_dict['rand'] = np.concatenate(
                (region_dict['rand'], region_density_dict['rand']),
                axis=1)
            region_dict['area'] = np.concatenate(
                (region_dict['area'], region_density_dict['area']),
                axis=1)
            region_dict['resolution'] = np.concatenate(
                (region_dict['resolution'], region_density_dict['resolution']),
                axis=1)

    return region_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_region_pickle_files', required=True,
                        type=str, help='Comma separated list of pickle '
                        'files containing the over-densities output by '
                        'pdf_maker.py. These files should contain the same '
                        'redshift binning.')
    parser.add_argument('--input_special_region_pickle_files', default=None,
                        type=str, help='Comma separated list of pickle '
                        'files containing the over-densities output by '
                        'pdf_maker.py. These files are distinct from above '
                        'as they contain specific regions that one always '
                        'wants to include. This could be for instance a '
                        'region of higher redshift but smaller area that you '
                        'want to include in the recovery but cannot combine '
                        'in the maker.')
    parser.add_argument('--special_region_weight', default=1.0,
                        type=float, help='Weight relative to non-special '
                        'regions. For instance if the special represent a '
                        'smaller area, this can be used to down weight them.')
    parser.add_argument('--output_pdf_file', required=True,
                        type=str, help='Name of ascii file to write '
                        'resultant pdf to.')
    parser.add_argument('--n_bootstrap', default=1000, type=int,
                        help='Argument specifying the number of bootstrap '
                        'resamlings of the recovery to compute errors.')
    parser.add_argument('--output_bootstraps_file', default=None, type=str,
                        help='This is an optional argument specifying an '
                        'ascii file to write the individual bootstrap pdfs '
                        'to.')
    parser.add_argument('--bootstrap_samples', default=None, type=str,
                        help='This is an optional argument specifying an '
                        'ascii file containing specified bootstrap samplings '
                        'to run. These should row-wise specifications of '
                        'regions from the input pair hdf5 file. Overrides '
                        'the number set in n_bootstrap.')

    args = parser.parse_args()

    # Load the input pickles.
    file_name_list = args.input_region_pickle_files.split(',')
    region_dict = load_from_pickle(file_name_list)
    # If we want a treat a set of data specially in the bootstrapping process
    # we load it here.
    if args.input_special_region_pickle_files is not None:
        file_name_list = args.input_special_region_pickle_files.split(',')
        region_special_dict = load_from_pickle(file_name_list)

    # Create the array of indices for the regions we will bootstrap over.
    if args.bootstrap_samples is None:
        bootstrap_samples = np.random.randint(
            region_dict['n_regions'],
            size=(args.n_bootstrap, region_dict['n_regions']))
        # Create the bootstraps for the "special" sample and concatenate them
        # to the end of the bootstrap samples.
        if args.input_special_region_pickle_files is not None:
            bootstrap_samples = np.concatenate(
                (bootstrap_samples,
                 np.random.randint(
                     region_special_dict['n_regions'],
                     size=(args.n_bootstrap,
                           region_special_dict['n_regions']))), axis=1)
    # If requested, the code can load a set of fixed bootstraps from disc.
    # If using a "special" sample make sure the bootstraps are formated as
    # above with the region ids appended to the end of the "normal" regions.
    else:
        bootstrap_samples = np.loadtxt(args.bootstrap_samples,
                                       dtype=np.int_)
        args.n_bootstrap = bootstrap_samples.shape[0]
    # Create empty array for storage of the bootstraps.
    density_bootstrap_array = np.empty((region_dict['redshift'].shape[0],
                                        args.n_bootstrap))

    # Computing mean redshift per bin.
    redshift_array = np.sum(region_dict['redshift'], axis=1)
    n_reference_array = np.sum(region_dict['n_reference'], axis=1)
    if args.input_special_region_pickle_files is not None:
        redshift_array += (args.special_region_weight *
                           np.sum(region_special_dict['redshift'], axis=1))
        n_reference_array += (
            args.special_region_weight *
            np.sum(region_special_dict['n_reference'], axis=1))
    redshift_array /= n_reference_array

    # Start the actual bootstrap process.
    for boot_idx, boot_reg_ids in enumerate(bootstrap_samples):

        tmp_boot_reg_ids = boot_reg_ids[:region_dict['n_regions']]
        boot_unknown_array = np.sum(
            region_dict['unknown'][:, tmp_boot_reg_ids], axis=1)
        boot_rand_array = np.sum(region_dict['rand'][:, tmp_boot_reg_ids],
                                 axis=1)
        # Compute the bootstrap average for the "special" samples.
        if args.input_special_region_pickle_files is not None:
            n_special_region = 0
            tmp_boot_reg_ids = boot_reg_ids[
                region_dict['n_regions'] + n_special_region:
                region_dict['n_regions'] + n_special_region +
                region_special_dict['n_regions']]
            n_special_region += region_special_dict['n_regions']
            boot_unknown_array += args.special_region_weight * np.sum(
                region_special_dict['unknown'][:, tmp_boot_reg_ids],
                axis=1)
            boot_rand_array += args.special_region_weight * np.sum(
                region_special_dict['rand'][:, tmp_boot_reg_ids], axis=1)
        # Compute the over density for the current bootstrap.
        density_bootstrap_array[:, boot_idx] = (boot_unknown_array /
                                                boot_rand_array - 1.0)

    # Compute the mean and standard deviation using nan safe means and
    # variances.
    density_array = np.nanmean(density_bootstrap_array, axis=1)
    density_err_array = np.nanstd(density_bootstrap_array, axis=1)

    # Create the output ascii header we will use to store the information on
    # this run.
    output_header = 'input_flags:\n'
    for arg in vars(args):
        output_header += '\t%s : %s\n' % (arg, getattr(args, arg))

    # If requested output the individual bootstraps to a file.
    if args.output_bootstraps_file is not None:
        np.savetxt(args.output_bootstraps_file, density_bootstrap_array,
                   header=output_header)

    # Add the column names to the header.
    output_header += ("type1 = z_mean\n"
                      "type2 = phi(z)\n"
                      "type3 = phi_err(z)\n")

    # Write the output.
    np.savetxt(args.output_pdf_file,
               np.array([redshift_array, density_array,
                         density_err_array]).transpose(),
               header=output_header)
