### TODO:
###     split core into two with one having the stomp calls and one not.

from astropy.cosmology import Planck13
from astropy.io import fits
import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iu_spline
# import stomp
import sys

verbose = False
_initialized_cosmology = False
_comov_dist_to_redshift_spline = None


def set_verbose(v_bool):
    verbose = v_bool
    return None   

def _initialize_cosmology():
    
    redshift_array = np.linspace(0.0, 10.0, 1000)
    comov_array = Planck13.comoving_distance(redshift_array)
    _comov_dist_to_redshift_spline = iu_spline(comov_array, redshift_array)

def redshift(z):
    if not _initialized_cosmology:
        _initialize_cosmology()
    return _comov_dist_to_redshift_spline(z)

def auto_region_resolution(n_obj, area):
    """
    Copied from STOMP::AngularCorrelation
    """
    max_resolution = 2048;
    if area > 500.0:
        ### large survey limit
        max_resolution = 512;
        if n_obj < 500000: max_resolution = 64
        if n_obj > 500000 and n_obj < 2000000: max_resolution = 128
        if n_obj > 2000000 and n_obj < 10000000: max_resolution = 256
    else: 
        ### small survey limit
        if n_obj < 500000: max_resolution = 256
        if n_obj > 500000 and n_obj < 2000000: max_resolution = 512;
        if n_obj > 2000000 and n_obj < 10000000: max_resolution = 1024;
        
    return max_resolution

def file_checker_loader(sample_file_name):
    
    try:
        file_handle = open(sample_file_name)
        file_handle.close()
    except IOError:
        print("IOError: File %s not found. The WiZZ is exiting.")
        raise IOError
    
    data_type = sample_file_name.split('.')[-1]
    if data_type == 'fit' or data_type == 'fits' or data_type == 'cat':
        
        hdu_list = fits.open(sample_file_name)
        data = hdu_list[1].data
        hdu_list.close()
        return data
    else:
        print("File type not currently supported. Try again later. "
              "The WiZZ is exiting.")
        raise IOError
    
    return None
    
def load_unknown_sample(sample_file_name, stomp_map, args):
    
    print("Loading unknown sample...")
    
    sample_data = file_checker_loader(sample_file_name)
    
    unknown_itree_map = stomp.IndexedTreeMap(stomp_map.RegionResolution(), 200)

    for idx, obj in enumerate(sample_data):
        tmp_iang = stomp.IndexedAngularCoordinate(
            obj[args.target_ra_name], obj[args.target_dec_name], idx,
            stomp.AngularCoordinate.Equatorial)
        if args.unknown_index_name is not None:
            tmp_iang.SetIndex(int(obj[args.unknown_index_name]))
        if stomp_map.Contains(tmp_iang):
            unknown_itree_map.AddPoint(tmp_iang)
    
    print("\tLoaded %i / %i target galaxies..." %
          (unknown_itree_map.NPoints(), sample_data.shape[0]))
    return unknown_itree_map

def load_target_sample(sample_file_name, stomp_map, args):
    
    print("Loading target sample...")
    
    sample_data = file_checker_loader(sample_file_name)
        
    target_vect = stomp.CosmoVector()
    target_idx_array = np.ones(sample_data.shape[0]) * -99
    for idx, obj in enumerate(sample_data):
        if (obj[args.target_redshift_name] < args.z_min or
            obj[args.target_redshift_name] >= args.z_max):
            continue
        tmp_cang = stomp.CosmoCoordinate(
            np.double(obj[args.target_ra_name]),
            np.double(obj[args.target_dec_name]),
            np.double(obj[args.target_redshift_name]), 1.0,
            stomp.AngularCoordinate.Equatorial)
        if stomp_map.Contains(tmp_cang):
            target_vect.push_back(tmp_cang)
            if args.target_index_name is None:
                target_idx_array[idx] = idx
            else:
                target_idx_array[idx] = obj[args.target_index_name]
    print("\tLoaded %i / %i target galaxies..." %
          (target_vect.size(), sample_data.shape[0]))
    return target_vect, target_idx_array[target_idx_array > -99]

def create_random_data(n_randoms, stomp_map):
    
    print("Creating %i n_randoms..." % n_randoms)
    random_vect = stomp.AngularVector()
    stomp_map.GenerateRandomPoints(random_vect, n_randoms)
        
    random_tree = stomp.TreeMap(stomp_map.RegionResolution(), 200)
        
    print("\tLoading randoms into tree map...")
    for rand_ang in random_vect:
        random_tree.AddPoint(rand_ang, 1.0)
        
    return random_tree


def create_hdf5_file(hdf5_file_name, args):
    
    hdf5_file = h5py.File(hdf5_file_name, 'w-', libver = 'latest')
    
    return hdf5_file

def load_pair_hdf5(hdf5_file_name):
    
    hdf5_file = h5py.File(hdf5_file_name, 'r')
    
    return hdf5_file

def close_hdf5_file(hdf5_file):
    
    hdf5_file.close()
    
    return None
    