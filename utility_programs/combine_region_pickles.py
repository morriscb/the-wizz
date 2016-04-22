
import argparse
import numpy as np
import pickle

"""
This program allows for the combination of different The-wiZZ, pdf_maker runs
after the fact using the region pickle files output from the code. This can be
useful for combining different recoveries from different spectroscoic pointings
in a way that is internally consistent.
"""

def load_from_pickle(file_name_list):
    
    region_dict = {'n_regions' : 0,
                   'redshift' : np.array([]),
                   'n_target' : np.array([]),
                   'unknown' : np.array([]),
                   'rand' : np.array([]),
                   'area' : np.array([]),
                   'resolution' : np.array([])}
    
    for file_idx, file_name in enumerate(file_name_list):
        
        pkl_file = open(file_name)
        region_density_dict = pickle.load(pkl_file)
        pkl_file.close()
        
        region_dict['n_regions'] += region_density_dict['n_regions']
        if file_idx == 0:
            region_dict['redshift'] = region_density_dict['redshift']
            region_dict['n_target'] = region_density_dict['n_target']
            region_dict['unknown'] = region_density_dict['unknown']
            region_dict['rand'] = region_density_dict['rand']
            region_dict['area'] = region_density_dict['area']
            region_dict['resolution'] = region_density_dict['resolution']
        else:
            region_dict['redshift'] = np.concatenate(
                (region_dict['redshift'], region_density_dict['redshift']),
                axis = 1)
            region_dict['n_target'] = np.concatenate(
                (region_dict['n_target'], region_density_dict['n_target']),
                axis = 1)
            region_dict['unknown'] = np.concatenate(
                (region_dict['unknown'], region_density_dict['unknown']),
                axis = 1)
            region_dict['rand'] = np.concatenate(
                (region_dict['rand'], region_density_dict['rand']),
                axis = 1)
            region_dict['area'] = np.concatenate(
                (region_dict['area'], region_density_dict['area']),
                axis = 1)
            region_dict['resolution'] = np.concatenate(
                (region_dict['resolution'], region_density_dict['resolution']),
                axis = 1)
    
    return region_dict

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_region_pickle_files', required = True,
                        type = str, help = 'Comma separated list of pickle '
                        'files containing the over-densities output by '
                        'pdf_maker.py. These files should contain the same '
                        'redshift binning.')
    parser.add_argument('--input_special_region_pickle_files', default = None,
                        type = str, help = 'Comma separated list of pickle '
                        'files containing the over-densities output by '
                        'pdf_maker.py. These files are distinct from above '
                        'as they contain specific regions that one always '
                        'wants to include. This could be for instance a region '
                        'of higher redshift but smaller area that you want to '
                        'include in the recovery but cannot combine in the '
                        'maker.')
    parser.add_argument('--output_pdf_file', required = True,
                        type = str, help = 'Name of ascii file to write '
                        'resultant pdf to.')
    parser.add_argument('--n_bootstrap', default = 1000, type = int,
                        help = 'Argument specifying the number of bootstrap '
                        'resamlings of the recovery to compute errors.')
    parser.add_argument('--output_bootstraps_file', default = None, type = str,
                        help = 'This is an optional argument specifying an '
                        'ascii file to write the individual bootstrap pdfs to.')
    parser.add_argument('--bootstrap_samples', default = None, type = str,
                        help = 'This is an optional argument specifying an '
                        'ascii file containing specified bootstrap samplings '
                        'to run. These should row-wise specifications of '
                        'regions from the input pair hdf5 file. Overrides '
                        'the number set in n_bootstrap.')
    parser.add_argument('--bootstrap_special_samples', default = None,
                        type = str, help = 'This is an optional argument '
                        'specifying an ascii file containing specified '
                        'bootstrap samplings to run. These should row-wise '
                        'specifications of regions from the input pair hdf5 '
                        'file. Overrides the number set in n_bootstrap.')
    
    args = parser.parse_args()
    
    file_name_list = args.input_region_pickle_files.split(',')
    region_dict = load_from_pickle(file_name_list)
    if args.input_special_region_pickle_files is not None:
        file_name_list = args.input_special_region_pickle_files.split(',')
        region_special_dict = load_from_pickle(file_name_list)
        
    if args.bootstrap_samples is None:
        bootstrap_samples = np.empty((region_dict['redshift'].shape[0],
                                      args.n_bootstrap))
    else:
        bootstrap_samples = np.loadtxt(args.bootstrap_samples)
        args.n_bootstrap = bootstrap_samples.shape[0]
        
    redshift_array = np.sum(region_dict['redshift'], axis = 1)
    n_target_array = np.sum(region_dict['n_target'], axis = 1)
    if args.input_special_region_pickle_files is not None:
        redshift_array += np.sum(region_special_dict['redshift'], axis = 1)
        n_target_array += np.sum(region_special_dict['n_target'], axis = 1)
    redshift_array /= n_target_array
        
    for boot_idx in xrange(args.n_bootstrap):
        
        boot_reg_ids = np.random.randint(region_dict['n_regions'],
                                         size = region_dict['n_regions'])
        boot_unknown_array = np.sum(region_dict['unknown'][:,boot_reg_ids],
                                    axis = 1)
        boot_rand_array = np.sum(region_dict['rand'][:,boot_reg_ids],
                                 axis = 1)
        if args.input_special_region_pickle_files is not None:
            boot_reg_ids = np.random.randint(
                region_special_dict['n_regions'],
                size = region_special_dict['n_regions'])
            boot_unknown_array += np.sum(
                region_special_dict['unknown'][:,boot_reg_ids], axis = 1)
            boot_rand_array += np.sum(
                region_special_dict['rand'][:,boot_reg_ids], axis = 1)
        bootstrap_samples[:, boot_idx] = (boot_unknown_array / boot_rand_array -
                                          1.0)
    
    density_array = np.nanmean(bootstrap_samples, axis = 1)
    density_err_array = np.nanstd(bootstrap_samples, axis = 1)
    
    np.savetxt(args.output_pdf_file,
               np.array([redshift_array, density_array,
                         density_err_array]).transpose())
    
        
        
        
        
        
        
        
        
    