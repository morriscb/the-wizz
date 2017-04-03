
from __future__ import division, print_function, absolute_import

import unittest

import stomp

from the_wizz import stomp_utils


class DummyArgs(object):

    def __init__(self):
        self.reference_ra_name = 'ra'
        self.reference_dec_name = 'dec'
        self.reference_redshift_name = 'z'
        self.reference_index_name = None

        self.z_min = 0.01
        self.z_max = 10.0

        self.unknown_ra_name = 'ra'
        self.unknown_dec_name = 'dec'
        self.unknown_index_name = 'id'


class TestStompUtils(unittest.TestCase):

    def setUp(self):
        self.stomp_map = stomp.Map(
            'data/COSMOS_X_zCOSMOS_BRIGHT_excluded.map')
        self.stomp_map.InitializeRegions(8)
        self.reference_cat_name = (
            'data/zCOSMOS_BRIGHT_v3.5_spec_FLAG34_FLAG134.fits')
        self.unknown_cat_name = (
            'data/COSMOS_iband_2009_radecidstomp_regionzp_best.fits')
        self.dummy_args = DummyArgs()

    def test_load_unknown_sample(self):
        unknown_itree_map = stomp_utils.load_unknown_sample(
            self.unknown_cat_name, self.stomp_map,
            self.dummy_args)

        self.assertEqual(244265, unknown_itree_map.NPoints())

    def test_load_reference_sample(self):
        (reference_vect, reference_id_array,
         reference_tree) = stomp_utils.load_reference_sample(
            self.reference_cat_name, self.stomp_map,
            self.dummy_args)

        self.assertEqual(5664, reference_vect.size())
        self.assertEqual(reference_vect.size(), reference_tree.NPoints())

    def test_create_random_data(self):
        random_tree = stomp_utils.create_random_data(10000, self.stomp_map)

        self.assertEqual(10000, random_tree.NPoints())
        self.assertEqual(10000, random_tree.Weight())


if __name__ == '__main__':

    unittest.main()