
from __future__ import division, print_function, absolute_import

import numpy as np
import subprocess
import unittest

from astropy.io import fits
import h5py
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
        self.unknown_weight_name = None
        self.unknown_stomp_region_name = 'stomp_region'

        self.use_inverse_weighting = True
        self.n_processes = 2
        self.n_reference_load_size = 10000
        self.use_reference_cleaning = False

class TestPDFMakerUtils(unittest.TestCase):

    def setUp(self):
        self.dummy_args = DummyArgs()

    def tearDown(self):
        pass

    def test_create_linear_redshift_bin_edges(self):
        z_array = pdf_maker_utils._create_linear_redshift_bin_edges(
            self.dummy_args.z_min, self.dummy_args.z_max,
            self.dummy_args.n_z_bins)
        print(z_array)

    def test_create_adaptive_redshift_bin_edges(self):
        pass

    def test_create_logspace_redshift_bin_edges(self):
        pass

    def test_create_comoving_redshift_bin_edges(self):
        pass

    def test_make_redshift_spline(self):
        pass

    def test_collapse_ids_to_single_estimate(self):

        cosmos_data = fits.getdata(
            'data/COSMOS_iband_2009_radecidstomp_regionzp_best.fits')
        cosmos_zp_cut = cosmos_data[np.logical_and(
            cosmos_data.zp_best > 0.3, cosmos_data.zp_best <= 0.5)]

        pdf_maker_obj = pdf_maker_utils.collapse_ids_to_single_estimate(
            self.dummy_args.input_pair_hdf5_file,
            self.dummy_args.pair_scale_name, cosmos_zp_cut,
            self.dummy_args)

        open_hdf5_file = h5py.File(self.dummy_args.input_pair_hdf5_file)
        hdf5_data_grp = open_hdf5_file['data']

        self.assertEqual(len(pdf_maker_obj.reference_redshift_array),
                         len(hdf5_data_grp))

        tmp_output_file = open('data/unittest_pdf_maker_raw.ascii')
        for reference_idx, key_name in enumerate(hdf5_data_grp):

            self.assertAlmostEqual(
                pdf_maker_obj.reference_redshift_array[reference_idx],
                hdf5_data_grp[key_name].attrs['redshift'], places=6)
            self.assertEqual(
                pdf_maker_obj.reference_region_array[reference_idx],
                hdf5_data_grp[key_name].attrs['region'])

            self.assertAlmostEqual(
                pdf_maker_obj.reference_area_array[reference_idx],
                hdf5_data_grp['%s/%s' %
                              (key_name,
                                self.dummy_args.pair_scale_name)
                              ].attrs['area'])
            self.assertEqual(
                pdf_maker_obj.reference_resolution_array[reference_idx],
                hdf5_data_grp['%s/%s' %
                              (key_name,
                                self.dummy_args.pair_scale_name)
                              ].attrs['bin_resolution'])

            ref_row_list = tmp_output_file.readline().split(' ')
            ref_unkn = np.float32(ref_row_list[0])
            self.assertAlmostEqual(
                pdf_maker_obj.reference_unknown_array[reference_idx],
                ref_unkn)
            ref_ref_den = np.float32(ref_row_list[1])
            self.assertAlmostEqual(
                pdf_maker_obj.reference_unknown_array[reference_idx],
                ref_ref_den)
            ref_hold_rand = np.float32(ref_row_list[2])
            self.assertAlmostEqual(
                pdf_maker_obj.reference_hold_rand_array[reference_idx],
                ref_hold_rand)
            ref_rand = np.float32(ref_row_list[3])
            if ref_rand > 0:
                self.assertAlmostEqual(
                    pdf_maker_obj.reference_rand_array[reference_idx] /
                    ref_rand - 1, 0.0, places=6)
            else:
                self.assertAlmostEqual(
                    pdf_maker_obj.reference_rand_array[reference_idx],
                    0.0, places=6)
        tmp_output_file.close()

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

if __name__ == '__main__':

    unittest.main()
