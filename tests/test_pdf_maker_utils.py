
from __future__ import division, print_function, absolute_import

import subprocess
import unittest

from astropy.io import fits
import stomp

from the_wizz import core_utils
from the_wizz import pdf_maker_utils


class DummyArgs(object):

    def __init__(self):
        self.input_pair_hdf5_file = 'data/unittest_output.hdf5'
        self.pair_scale_name = 'kpc100t300'
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

    def test_create_adaptive_redshift_bin_edges(self):
        pass

    def test_create_logspace_redshift_bin_edges(self):
        pass

    def test_create_comoving_redshift_bin_edges(self):
        pass

    def test_make_redshift_spline(self):
        pass

    def test_collapse_ids_to_single_estimate(self):
        pass

    def test_load_pair_data(self):
        pass

    def test_collapse_full_sample(self):
        pass

    def test_pdf_maker_creation_and_load(self):
        pass

    def test_pdf_maker_reset(self):
        pass

    def test_scale_random_points(self):
        pass

    def test_compute_region_densities(self):
        pass

    def test_compute_pdf(self):
        pass

    def test_compute_pdf_bootstrap(self):
        pass

    def test_write_bootstrap_samples_to_ascii(self):
        pass

    def test_write_pdf_to_ascii(self):
        pass
