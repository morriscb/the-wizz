
"""Utility functions for creating and communicating with the STOMP spherical
pixelization library. I set these files aside so that the part of the code that
combines the pairs given a pair file and indexes is usable without having the
STOMP library compiled.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import stomp

import _core_utils


def load_unknown_sample(sample_file_name, stomp_map, args):
    """ Method for loading a set of objects with unknown redshifts into
    The-wiZZ. This function maskes the data and returns a STOMP.iTreeMap object
    which is a searchable quad tree where each object stored has a unique index.
    If a name of an index column is not specified a simple counting index from
    thestart of the file is stored.
    ----------------------------------------------------------------------------
    Args:
        sample_file_name: string name specifying the file containing the unknown
            sample. Assumed file type is FITS.
        stomp_map: STOMP.Map object specifying the geomometry of the area
            considered.
        args: ArgumentParser.parse_args object returned from input_flags.
    Returns:
        a STOMP::iTree map object
    """
    print("Loading unknown sample...")
    # TODO: This is the main bottle neck of the code. Need to make loading,
    #     masking, and creating the quadtree much faster. This may require
    #     creating a python wrapped C++ function for loading and creating
    #     a STOMP iTreeMap.
    sample_data = _core_utils.file_checker_loader(sample_file_name)
    unknown_itree_map = stomp.IndexedTreeMap(stomp_map.RegionResolution(), 200)
    for idx, obj in enumerate(sample_data):
        tmp_iang = stomp.IndexedAngularCoordinate(
            obj[args.unknown_ra_name], obj[args.unknown_dec_name], idx,
            stomp.AngularCoordinate.Equatorial)
        if args.unknown_index_name is not None:
            tmp_iang.SetIndex(int(obj[args.unknown_index_name]))
        if stomp_map.Contains(tmp_iang):
            unknown_itree_map.AddPoint(tmp_iang)
    print("\tLoaded %i / %i target galaxies..." %
          (unknown_itree_map.NPoints(), sample_data.shape[0]))
    return unknown_itree_map


def load_target_sample(sample_file_name, stomp_map, args):
    """Method for loading the targert object sample with known redshifts. The
    objects are masked against the requested geomometry and stored with their
    redshifts and into a STOMP.CosmoVector object. The code also returns an
    array of the indices of the target objects from the columns requested in
    input_flags or simply counts.
    ----------------------------------------------------------------------------
    Args:
        sample_file_name: string name of the file containing the target, known
            redshift objects. Currently only FITS is supported.
        stomp_map: STOMP.Map object specifying the geomometry of the area
            considered.
        args: ArgumentParser.parse_args object returned from input_flags.
    Returns:
        tuple: STOMP::CosmosVector, int array
    """
    print("Loading target sample...")
    sample_data = _core_utils.file_checker_loader(sample_file_name)
    target_vect = stomp.CosmoVector()
    target_idx_array = np.ones(sample_data.shape[0])*-99
    for idx, obj in enumerate(sample_data):
        if (obj[args.target_redshift_name] < args.z_min or
            obj[args.target_redshift_name] >= args.z_max):
            # Continue if the target object redshift is out of range.
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
    """Function for creating randomly positioned unknown objects on the considerd
    geomometry. These is used for normalizing the output PDF and properly
    estimating the "zero point" of the correlation amplitude. The code returns
    a spatially searchable quad tree of the random points.
    ----------------------------------------------------------------------------
    Args:
        n_randoms: int number of random points to generate
        stomp_map: STOMP.Map object specifying the survey geomometry
    Returns:
        STOMP::TreeMap object
    """
    print("Creating %i n_randoms..." % n_randoms)
    random_vect = stomp.AngularVector()
    stomp_map.GenerateRandomPoints(random_vect, n_randoms)
    random_tree = stomp.TreeMap(stomp_map.RegionResolution(), 200)
    print("\tLoading randoms into tree map...")
    for rand_ang in random_vect:
        random_tree.AddPoint(rand_ang, 1.0)
    return random_tree
