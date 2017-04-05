
from __future__ import division, print_function, absolute_import

import numpy as np
import h5py
import subprocess
import unittest

import stomp

from the_wizz import pair_maker_utils
from the_wizz import stomp_utils


class DummyArgs(object):

    def __init__(self):
        self.reference_ra_name = 'ra'
        self.reference_dec_name = 'dec'
        self.reference_redshift_name = 'z'
        self.reference_index_name = None

        self.unknown_ra_name = 'ra'
        self.unknown_dec_name = 'dec'
        self.unknown_index_name = 'id'

        self.z_min = 0.01
        self.z_max = 10.0

        self.n_randoms = 10
        self.output_hdf5_file = 'unittest_output.hdf5'


class TestPairMakerUtils(unittest.TestCase):

    def setUp(self):
        self.dummy_args = DummyArgs()
        self.stomp_map = stomp.Map(
            'data/COSMOS_X_zCOSMOS_BRIGHT_excluded.map')
        self.stomp_map.InitializeRegions(8)
        (self.reference_vect, self.reference_id_array,
         self.reference_tree) = stomp_utils.load_reference_sample(
            'data/zCOSMOS_BRIGHT_v3.5_spec_FLAG34_FLAG134.fits',
            self.stomp_map, self.dummy_args)
        self.unknown_tree = stomp_utils.load_unknown_sample(
            'data/COSMOS_iband_2009_radecidstomp_regionzp_best.fits',
            self.stomp_map,self.dummy_args)

    def tearDown(self):
        subprocess.Popen('rm unittest_output.hdf5', shell=True)

    def test_raw_pair_finder_and_hdf5_creation(self):
        pair_finder = pair_maker_utils.RawPairFinder(
            self.unknown_tree, self.reference_vect,
            self.reference_id_array, self.reference_tree,
            self.stomp_map, self.dummy_args.output_hdf5_file, None,
            create_hdf5_file=True, input_args=self.dummy_args)

    def test_pair_maker_output(self):

        pair_finder = pair_maker_utils.RawPairFinder(
            self.unknown_tree, self.reference_vect, self.reference_id_array,
            self.reference_tree, self.stomp_map,
            self.dummy_args.output_hdf5_file, None, create_hdf5_file=True,
            input_args=self.dummy_args)
        pair_finder.find_pairs(100, 300)

        output_hdf5_file = h5py.File(self.dummy_args.output_hdf5_file, 'r')
        test_hdf5_file = h5py.File('data/test_COSMOS_pair_data.hdf5')

        for reference_idx, reference_obj in enumerate(self.reference_vect):
            ref_grp = output_hdf5_file[
                'data/%i' %
                self.reference_id_array[reference_idx]]
            scale_grp = output_hdf5_file[
                'data/%i/kpc100t300' %
                self.reference_id_array[reference_idx]]
            ref_pair_id_array = scale_grp['ids'][...]
            ref_dist_weight_array = scale_grp['dist_weights'][...]

            test_grp = test_hdf5_file['kpc100t300/%i' %
                                      self.reference_id_array[reference_idx]]
            test_pair_id_array = test_grp['ids'][...]
            test_dist_weight_array = test_grp['inv_dist'][...]
            if len(ref_pair_id_array) != len(test_pair_id_array):
                print('Failed for reference id:',
                      self.reference_id_array[reference_idx])
            self.assertEqual(len(ref_pair_id_array), len(test_pair_id_array))
            for ref_pair_id, test_pair_id, \
              ref_dist_weight, test_dist_weight in \
              zip(ref_pair_id_array, test_pair_id_array,
                  ref_dist_weight_array, test_dist_weight_array):
                self.assertEqual(ref_pair_id, test_pair_id)
                self.assertAlmostEqual(ref_dist_weight, test_dist_weight)
            self.assertAlmostEqual(
                ref_grp.attrs['redshift'],
                test_grp.attrs['redshift'])
            self.assertAlmostEqual(
                scale_grp.attrs['unmasked_frac'],
                test_grp.attrs['unmasked_frac'])
            self.assertEqual(
                scale_grp.attrs['bin_resolution'],
                test_grp.attrs['bin_resolution'])
            self.assertAlmostEqual(
                scale_grp.attrs['area'],
                test_grp.attrs['area'])

    def test_pair_maker_with_randoms(self):

        random_tree = stomp_utils.create_random_data(
            self.dummy_args.n_randoms * self.unknown_tree.NPoints(),
            self.stomp_map)

        pair_finder = pair_maker_utils.RawPairFinder(
            self.unknown_tree, self.reference_vect, self.reference_id_array,
            self.reference_tree, self.stomp_map,
            self.dummy_args.output_hdf5_file, random_tree,
            create_hdf5_file=True, input_args=self.dummy_args)
        pair_finder.find_pairs(100, 300)

        output_hdf5_file = h5py.File(self.dummy_args.output_hdf5_file, 'r')
        data_grp = output_hdf5_file['data']
        n_random = data_grp.attrs['n_random']
        tot_area = data_grp.attrs['area']

        ref_random_sum = 0
        ref_area_sum = 0.
        for reference_idx, reference_obj in enumerate(self.reference_vect):
            scale_grp = output_hdf5_file[
                'data/%i/kpc100t300' %
                self.reference_id_array[reference_idx]]
            ref_random_sum += scale_grp.attrs['n_random']
            ref_area_sum += scale_grp.attrs['area']
        self.assertAlmostEqual(
            (n_random / tot_area) /
            (ref_random_sum / ref_area_sum) - 1,
            0.00, places=2)
