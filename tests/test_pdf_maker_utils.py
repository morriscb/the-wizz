
from __future__ import division, print_function, absolute_import

import subprocess
import unittest

import stomp

from the_wizz import pdf_maker_utils


class DummyArgs(object):

    def __init__(self):
        self.reference_ra_name = 'ra'
        self.reference_dec_name = 'dec'
        self.reference_redshift_name = 'z'
        self.reference_index_name = None
        self.z_min = 0.01
        self.z_max = 10.0
        self.n_z_bins = 10

        self.unknown_ra_name = 'ra'
        self.unknown_dec_name = 'dec'
        self.unknown_index_name = 'id'

class TestPDFMakerUtils(unittest.TestCase):

    def setUp(self):
        self.dummy_args = DummyArgs()

    def tearDown(self):
        pass

    def test_create_linear_redshift_bin_edges(self):
        z_array = pdf_maker_utils._create_linear_redshift_bin_edges(
            self.dummy_args.z_min, self.dummy_args.z_max,
            self.dummy_args.n_z_bins)
