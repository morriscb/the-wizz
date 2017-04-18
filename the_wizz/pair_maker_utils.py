
"""Utility functions for finding and storing close pairs between a reference
object with known redshifts and objects with unknown redshifts.
"""

from __future__ import division, print_function, absolute_import

import h5py
from multiprocessing import Pool
import numpy as np

import stomp

from the_wizz.core_utils import create_hdf5_file


def _multi_proc_write_reference_to_hdf5(input_tuple):
    """
    Stores the resultant pairs to a requested HDF5 file. This function is
    for use in a async multiprocessing sub process.
    --------------------------------------------------------------------------
    """

    # Grab all the data we need to write to the file.
    (hdf5_file_name, ref_id, scale_name, id_array,
     dist_weight_array, bin_resolution, unmasked_frac, area,
     ref_ref_n_points, n_random, rand_dist_weight) = input_tuple

    # Open our file and load the group corresponding to the reference object
    # we are writing data for.
    open_hdf5_file = h5py.File(hdf5_file_name, 'a')
    ref_scale_grp = open_hdf5_file.create_group(
        'data/%i/%s' % (ref_id, scale_name))

    # Write the attribute data for this reference, this scale.
    ref_scale_grp.attrs.create('bin_resolution', bin_resolution)
    ref_scale_grp.attrs.create('unmasked_frac', unmasked_frac)
    ref_scale_grp.attrs.create('area', area)
    ref_scale_grp.attrs.create('ref_ref_n_points', ref_ref_n_points)

    # If we have randoms store those too.
    if n_random is not None:
        ref_scale_grp.attrs.create('n_random', n_random)
        ref_scale_grp.attrs.create('rand_dist_weight', rand_dist_weight)

    # Test of the array is empty so we can write an unbound shape in HDF5,
    # else set the correct size for the arrays.
    if id_array.shape[0] <= 0:
        tmp_max_shape = (None,)
    else:
        tmp_max_shape = id_array.shape

    # Sort the arrays on id and write to disk.
    sorted_args_array = id_array.argsort()
    ref_scale_grp.create_dataset(
        'ids', data=id_array[sorted_args_array],
        maxshape=tmp_max_shape, compression='lzf', shuffle=True)
    ref_scale_grp.create_dataset(
        'dist_weights', data=dist_weight_array[sorted_args_array],
        maxshape=tmp_max_shape, compression='lzf', shuffle=True)

    # Close the file.
    del ref_scale_grp
    open_hdf5_file.close()
    del open_hdf5_file

    return None


class RawPairFinder(object):
    """Main class for calculating and storing the indexes of nearby pairs for
    the reference and unknown samples. It handles both the real data and random
    samples in slightly different pair loops since we don't need to store
    indices for random samples.
    """
    def __init__(self, unknown_itree, reference_vector, reference_ids,
                 reference_tree, stomp_map, hdf5_file_name,
                 random_tree=None, create_hdf5_file=True, input_args=None):
        """Initialization for the pair finding software. Utilizes the STOMP
        sphereical pixelization library to find and store all close pairs into
        a HDF5 data file.
        ------------------------------------------------------------------------
        args:
            unknown_itree: stomp IndexedTreeMap object containing the unknown
                object sample. The data structure is that of a searchable
                quad-tree
            reference_vector: stomp CosmoVector object containing the reference
                objects
            reference_ids: numpy.array containing the index number of the
                reference objects
            reference_tree: stomp TreeMap object containing the reference
                sample.
            stomp_map: stomp Map object specifying the geometry of the survey.
            hdf5_file_name: string name of the HDF5 file to write the data out
                to.
            random_tree: stomp TreeMap object containing random points drawn
            create_hdf5_file: bool (default True) value specifiying if we need
                to create a new HDF5 output file. If not we skip the creation
                # step and expect to appen to a previously created file.
        """

        # Move in our declaed objects.
        self._unknown_itree = unknown_itree
        self._reference_vect = reference_vector
        self._reference_ids = reference_ids
        self._reference_tree = reference_tree
        self._stomp_map = stomp_map
        self._hdf5_file_name = hdf5_file_name
        self._random_tree = random_tree

        # If we need to, create and store all of the global stomp map
        # and reference data properties.
        if create_hdf5_file:
            self._create_hdf5_file_and_store_reference_data(
                input_args)

    def _create_hdf5_file_and_store_reference_data(self, input_args):
        """ Creates a initial HDF5 data file for use in the-wizz. This
        is where all of the final pair data will be stored after find pairs
        is run.
        """

        # Create a new hdf5 file
        hdf5_file = create_hdf5_file(self._hdf5_file_name, input_args)

        # Create a group called data where the pair data will go.
        data_grp = hdf5_file.create_group('data')

        # Find the area of the stomp map and each of its regions.
        region_area = np.empty(self._stomp_map.NRegion(), dtype=np.float32)
        for reg_idx in xrange(self._stomp_map.NRegion()):
            region_area[reg_idx] = self._stomp_map.RegionArea(reg_idx)

        # Store global numbers for this stomp_map that are not dependent
        # on the annulus we are measuring the clustering-zs in.
        data_grp.attrs.create('area', self._stomp_map.Area())
        data_grp.attrs.create('n_region', self._stomp_map.NRegion())
        data_grp.attrs.create('region_area', region_area)
        data_grp.attrs.create('n_unknown', self._unknown_itree.NPoints())
        if input_args.n_randoms > 0:
            data_grp.attrs.create('n_random_points',
                                  self._unknown_itree.NPoints() *
                                  input_args.n_randoms)

        # Loop over the reference objects.
        for reference_idx, reference_obj in enumerate(self._reference_vect):

            # Create the group where for this individual reference object
            # were we will store all the pair data.
            ref_grp = data_grp.create_group(
                '%s' % self._reference_ids[reference_idx])
            # Store information that is unique to each reference object.
            ref_grp.attrs.create('redshift', reference_obj.Redshift())
            ref_grp.attrs.create(
                'region', self._stomp_map.FindRegion(reference_obj))

        # Close the hdf5 file after we are done.
        hdf5_file.close()

        return None

    def find_pairs(self, min_scale, max_scale):
        """Main functionality of the RawPairFinder class. Given the input data,
        we find all the close pairs between the reference, known redshift
        objects and the unknown redshift oobjects. Stores the raw pair indices
        and also the inverse distance weight.
        ------------------------------------------------------------------------
        Args:
            min_scale: float value of the minimum scale to run the pair finder.
                Units are physical kpc
            max_scale: float value of the maximum scale to run the pair finder.
                Units are physical kpc
        Returns:
           None
        """

        # Create a multiprocessing pool to allow writing to disk instead of
        # storing everything in memory.
        hdf5_writer_pool = Pool(1)
        scale_name = 'kpc%it%i' % (min_scale, max_scale)

        # Create radial bin object that we will use to search the trees.
        radial_bin = stomp.RadialBin(min_scale / 1000.0, max_scale / 1000.0,
                                     0.01)

        # Start our loop over reference objects.
        print("Finding real and random pairs...")
        for reference_idx, reference_obj in enumerate(self._reference_vect):

            # Find the stomp region where this reference object resides.
            region_id = self._stomp_map.FindRegion(
                reference_obj)

            # Scale radial bin to correct on sky size given the redshift.
            radial_bin.SetRedshift(reference_obj.Redshift())
            max_ang = stomp.Cosmology.ProjectedAngle(reference_obj.Redshift(),
                                                     max_scale / 1000.0)

            # Find the most efficient stomp resolution for this annulus.
            radial_bin.CalculateResolution(reference_obj.Lambda() - max_ang,
                                           reference_obj.Lambda() + max_ang)
            reference_pix = stomp.Pixel(reference_obj, radial_bin.Resolution())

            # Find pixels that cover the annulus at a fixed resoulution
            covering_pix_vect = stomp.PixelVector()
            reference_pix.WithinAnnulus(
                reference_obj, radial_bin.Resolution(), radial_bin,
                covering_pix_vect)
            bin_resolution = radial_bin.Resolution()

            # Create the lists and values where we will put our out data
            # including the id and dist_weight arrays.
            output_id_list = []
            output_dist_weight_list = []
            unmasked_frac = 0.
            area = 0.
            ref_ref_n_points = 0
            n_random = None
            rand_dist_weight = None
            if self._random_tree is not None:
                n_random = 0
                rand_dist_weight = 0.

            # Start loop over the pixels that cover the annulus.
            for pix in covering_pix_vect:

                # Get the weight of the current pixel.
                dist_weight = self._compute_dist_weight(
                    reference_obj.ProjectedRadius(pix.Ang()))

                # Check this pixel against all of our tree maps.
                (pix_id_list, pix_dist_weight_list,
                 pix_unmasked_frac, pix_area,
                 pix_ref_ref_n_points, pix_n_random, pix_rand_dist_weight) = \
                    self._find_pairs_in_pixel(pix, region_id, dist_weight)

                # Store the values for this pixel and continue the loop.
                output_id_list.extend(pix_id_list)
                output_dist_weight_list.extend(pix_dist_weight_list)
                unmasked_frac += pix_unmasked_frac
                area += pix_area
                ref_ref_n_points += pix_ref_ref_n_points
                if pix_n_random is not None:
                    n_random += pix_n_random
                    rand_dist_weight += pix_rand_dist_weight

            # If we found any actual pairs for this annulus we store
            # them. Else we just sent the empty arrays.
            if len(output_id_list) > 0:
                ref_pair_id_array = np.concatenate(output_id_list)
                ref_dist_weight_array = np.concatenate(output_dist_weight_list)
            else:
                ref_pair_id_array = np.array([], dtype=np.uint32)
                ref_dist_weight_array = np.array([], dtype=np.float32)

            # Create a temp list of the tuple we will send off to be stored
            # by our multiprocessing worker.
            multi_proc_list = [
                (self._hdf5_file_name, self._reference_ids[reference_idx],
                 scale_name, ref_pair_id_array, ref_dist_weight_array,
                 bin_resolution, unmasked_frac, area, ref_ref_n_points,
                 n_random, rand_dist_weight)]

            # Send off our data for storage.
            hdf5_writer_pool.apply_async(
                _multi_proc_write_reference_to_hdf5, multi_proc_list)

        # Close out our multiprocessing worker pool.
        hdf5_writer_pool.close()
        hdf5_writer_pool.join()

        return None

    def _find_pairs_in_pixel(self, pix, region_id, dist_weight):
        """ Compute the pairs between the reference object and a given
        pixel.
        """

        # Create our temp veriables to fill.
        unmasked_frac = 0.
        area = 0.
        ref_ref_n_points = 0
        n_random = None
        rand_dist_weight = None
        if self._random_tree is not None:
            n_random = 0
            rand_dist_weight = 0.

        # Check to see if the resolution is not larger than the
        # regionation resolution. If it is break the pixel into its
        # children and loop over them to properly check which region the
        # the pixel is in.
        if pix.Resolution() < self._stomp_map.RegionResolution():
            pix_vect = stomp.PixelVector()
            pix.SubPix(self._stomp_map.RegionResolution(),
                       pix_vect)

            # Empty lists for outputs.
            pair_id_list = []
            pair_dist_wieght_list = []

            # Loop over the sub pixels.
            for sub_pix in pix_vect:
                # Recurse this function to find the pairs in the sub pixel.
                (sub_pix_id_list, sub_pix_dist_weight_list,
                 sub_pix_unmasked_frac, sub_pix_area,
                 sub_pix_ref_ref_n_points, sub_pix_n_random,
                 sub_pix_rand_dist_weight) = \
                    self._find_pairs_in_pixel(sub_pix, region_id, dist_weight)

                # Store all of our output data.
                pair_id_list.extend(sub_pix_id_list)
                pair_dist_wieght_list.extend(sub_pix_dist_weight_list)
                unmasked_frac += sub_pix_unmasked_frac
                area += sub_pix_area
                ref_ref_n_points += sub_pix_ref_ref_n_points
                if self._random_tree is not None:
                    n_random += sub_pix_n_random
                    rand_dist_weight += sub_pix_rand_dist_weight

            # Return the results for this pixel.
            return (pair_id_list, pair_dist_wieght_list, unmasked_frac, area,
                    ref_ref_n_points, n_random, rand_dist_weight)
        else:
            # Check that the region of the pixel is the same as the reference.
            # If not return
            if self._stomp_map.FindRegion(pix) != region_id:
                return ([], [], 0., 0., 0, n_random, rand_dist_weight)

            # Check the facdtion of area available in this pixel given
            # the input stomp map. Compute the pixel area if we don't return.
            unmasked_frac = self._stomp_map.FindUnmaskedFraction(pix)
            if unmasked_frac <= 0.0:
                return ([], [], 0., 0., 0, n_random, rand_dist_weight)
            area = unmasked_frac * pix.Area(pix.Resolution())

            # Get the pair ids and distances around the unknown sample.
            (pair_array, pair_dist_weight_array) = \
                self._reference_to_unknown_pixel(
                    pix, dist_weight)
            ref_ref_n_points = self._reference_to_reference_pixel(pix)

            # Get the randoms in this pixel if we have them.
            n_random = None
            rand_dist_weight = None
            if self._random_tree is not None:
                n_random, rand_dist_weight = \
                    self._reference_to_random_pixel(pix, dist_weight)

            return ([pair_array], [pair_dist_weight_array], unmasked_frac,
                    area, ref_ref_n_points, n_random, rand_dist_weight)

    def _compute_dist_weight(self, dist):
        """Convienence function for computing the weighting of an unknown object
        as a function of projected physical Mpc from the reference object. This
        function is declared here for easy modification by intersted users.
        The normal behavior is inverse distance.
        ------------------------------------------------------------------------
        Args:
            dist: float array of distances in projected physical Mpc
        Rreturns:
            float array of weights
        """
        return np.where(dist > 1e-8, 1./dist, 1e8)

    def _reference_to_unknown_pixel(self, pix, dist_weight):
        """Internal class function for finding the number of unknown objects and
        their ids in a single stomp pixel.
        ------------------------------------------------------------------------
        Args:
            pix: stomp.Pixel object to compute the number of randoms in.
            dist_weight: float value
        Returns:
            None
        """
        # Create an empty vector to put the found unknown pairs in and fill
        # the vector.
        tmp_i_ang_vect = stomp.IAngularVector()
        self._unknown_itree.Points(tmp_i_ang_vect, pix)

        # Since we know how many points we need we reserve that amount
        # in our output arrays.
        output_id_array = np.empty(tmp_i_ang_vect.size(), dtype=np.uint32)
        output_dist_weight_array = np.empty(
            tmp_i_ang_vect.size(), dtype=np.float32)

        # Fill the output arrays.
        for ang_idx, i_ang in enumerate(tmp_i_ang_vect):
            output_id_array[ang_idx] = i_ang.Index()
            output_dist_weight_array[ang_idx] = dist_weight

        return output_id_array, output_dist_weight_array

    def _reference_to_reference_pixel(self, pix):
        """Internal class function for finding the number of randoms in a
        single stomp pixel.
        ------------------------------------------------------------------------
        Args:
            pix: stomp.Pixel object to compute the number of randoms in.
        Returns:
            None
        """
        return self._reference_tree.NPoints(pix)

    def _reference_to_random_pixel(self, pix, dist_weight):
        """Internal class function for finding the number of randoms in a
        single stomp pixel.
        ------------------------------------------------------------------------
        Args:
            pix: stomp.Pixel object to compute the number of randoms in.
            dist_weight: float value
        Returns:
            None
        """
        rand_n_points = self._random_tree.NPoints(pix)
        rand_dist_weight = rand_n_points * dist_weight
        return rand_n_points, rand_dist_weight
