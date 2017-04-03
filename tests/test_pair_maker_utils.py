
from __future__ import division, print_function, absolute_import

import subprocess
import unittest

import stomp

from the_wizz import pair_maker_utils
from the_wizz import stomp_utils


class DummyArgs(object):

    def __init__(self):
        self.target_ra_name = 'ra'
        self.target_dec_name = 'dec'
        self.target_redshift_name = 'z'
        self.target_index_name = None

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
        (self.target_vect, self.target_id_array,
         self.target_tree) = stomp_utils.load_target_sample(
            'data/zCOSMOS_BRIGHT_v3.5_spec_FLAG34_FLAG134.fits',
            self.stomp_map, self.dummy_args)

    def tearDown(self):
        pass

    def test_raw_pair_finder_creation(self):
        pair_finder = pair_maker_utils.RawPairFinder(
            None, self.target_vect, self.target_id_array, self.target_tree,
            self.stomp_map)
        pair_finder._reset_array_data()
        pair_finder._reset_random_data()

        self.assertEqual(pair_finder._area_array.shape[0],
                         self.target_vect.size())
        self.assertEqual(pair_finder._unmasked_array.shape[0],
                         self.target_vect.size())
        self.assertEqual(pair_finder._bin_resolution.shape[0],
                         self.target_vect.size())
        self.assertEqual(pair_finder._target_target_array.shape[0],
                         self.target_vect.size())

        self.assertEqual(len(pair_finder._pair_list), self.target_vect.size())
        self.assertEqual(len(pair_finder._pair_invdist_list),
                         self.target_vect.size())

    def test_pair_maker_output(self):
        unknown_tree = stomp_utils.load_unknown_sample(
            'data/COSMOS_iband_2009_radecidstomp_regionzp_best.fits',
            self.stomp_map,self.dummy_args)

        pair_finder = pair_maker_utils.RawPairFinder(
            unknown_tree, self.target_vect, self.target_id_array,
            self.target_tree, self.stomp_map)
        pair_finder.find_pairs(100, 300)

        print("Found Pairs")

        output_pair_file = open('saved_pairs.ascii', 'w')
        output_dist_weight_file = open('saved_dist_weights.ascii', 'w')
        output_data = open('saved_data.ascii', 'w')
        for target_idx, target_obj  in enumerate(self.target_vect):
            print("Writing target:", target_idx)
            pair_ids = ''
            pair_dists = ''
            print("\twriting pairs")
            for pair_id, pair_dist_weight in \
              zip(pair_finder._pair_list[target_idx],
                  pair_finder._pair_invdist_list[target_idx]):
                pair_ids += '%i ' % pair_id
                pair_dists += '%.8e ' % pair_dist_weight
            output_pair_file.writelines('%s\n' % pair_ids)
            output_dist_weight_file.writelines('%s\n' % pair_dists)
            print("\twriting data")
            output_data.writelines(
                '%.8e %.8e %i %.8e %i %.8e %.8e\n' %
                (target_obj.Redshift(),
                 pair_finder._unmasked_array[target_idx],
                 pair_finder._bin_resolution[target_idx],
                 pair_finder._area_array[target_idx],
                 pair_finder._region_ids[target_idx],
                 pair_finder._target_target_array[target_idx] /
                 pair_finder._area_array[target_idx],
                 pair_finder._target_target_array[target_idx] /
                 pair_finder._area_array[target_idx]))
        output_pair_file.close()
        output_dist_weight_file.close()
        output_data.close()

        test_pair_data = open('data/saved_pairs.ascii')
        test_data = open('data/saved_data.ascii')
        for target_idx, target_obj, data_compare, pair_data in \
          data_compare_list = data_compare.split('')
          zip(self.target_vect, test_data, test_pair_data):
            pair_ids = ''
            pair_dists = ''
            print("\twriting pairs")
            for pair_id, pair_dist_weight in \
              zip(pair_finder._pair_list[target_idx],
                  pair_finder._pair_invdist_list[target_idx]):
                pair_ids += '%i ' % pair_id
                pair_dists += '%.8e ' % pair_dist_weight
            self.assertEqual(pair_ids, pair_data_list[:-2])
            self.assertEqual(target_obj.Redshift(),
                             float(data_compare_list[0]))
            self.assertEqual(pair_finder._unmasked_array[target_idx],
                             float(data_compare_list[1]))
            self.assertEqual(pair_finder._bin_resolution[target_idx],
                             int(data_compare_list[2]))
            self.assertEqual(pair_finder._area_array[target_idx],
                             float(data_compare_list[3]))
            self.assertEqual(pair_finder._region_ids[target_idx],
                             int(data_compare_list[4]))
            self.assertEqual(
                pair_finder._target_target_array[target_idx] /
                pair_finder._area_array[target_idx],
                float(data_compare_list[5]))
            self.assertEqual(
                pair_finder._target_target_array[target_idx] /
                pair_finder._area_array[target_idx],
                float(data_compare_list[4]))

    def test_write_to_hdf5(self):
        pass
