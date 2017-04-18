
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

        self.unknown_sample_file = \
            'data/' \
            'COSMOS_iband_2009_radecidstomp_regionzp_best.fits'
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
        bin_edges = [0.01,  1.009, 2.008, 3.007, 4.006,
                     5.005, 6.004, 7.003, 8.002, 9.001]
        z_array = pdf_maker_utils._create_linear_redshift_bin_edges(
            self.dummy_args.z_min, self.dummy_args.z_max,
            self.dummy_args.n_z_bins)
        for bin_edge, test_bin_edge in zip(z_array, bin_edges):
            self.assertAlmostEqual(bin_edge, test_bin_edge)

    def test_create_adaptive_redshift_bin_edges(self):
        bin_edges = [0.015, 1.015, 2.015, 3.015, 4.015,
                     5.015, 6.015, 7.015, 8.015, 9.015]
        z_array = pdf_maker_utils._create_adaptive_redshift_bin_edges(
            self.dummy_args.z_min, self.dummy_args.z_max,
            self.dummy_args.n_z_bins, np.arange(0.015, 9.95, 0.1))
        for bin_edge, test_bin_edge in zip(z_array, bin_edges):
            self.assertAlmostEqual(bin_edge, test_bin_edge)

    def test_create_logspace_redshift_bin_edges(self):
        bin_edges = [0.01, 0.28241475, 0.62830455, 1.0674869, 1.62512446,
                     2.33316666, 3.23218029, 4.3736737, 5.82304794,
                     7.66334389]

        z_array = pdf_maker_utils._create_logspace_redshift_bin_edges(
            self.dummy_args.z_min, self.dummy_args.z_max,
            self.dummy_args.n_z_bins)
        for bin_edge, test_bin_edge in zip(z_array, bin_edges):
            self.assertAlmostEqual(bin_edge, test_bin_edge)

    def test_create_comoving_redshift_bin_edges(self):
        bin_edges = [0.01, 0.24919755, 0.52181597, 0.84405631, 1.2392455,
                     1.74220817, 2.40724861, 3.32350705, 4.64656624,
                     6.66935238]
        z_array = pdf_maker_utils._create_comoving_redshift_bin_edges(
            self.dummy_args.z_min, self.dummy_args.z_max,
            self.dummy_args.n_z_bins)
        for bin_edge, test_bin_edge in zip(z_array, bin_edges):
            self.assertAlmostEqual(bin_edge, test_bin_edge)

    def test_collapse_ids_to_single_estimate(self):

        cosmos_data = fits.getdata(self.dummy_args.unknown_sample_file)
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

        tmp_output_file = open('data/unittest_pdf_maker_raw_data.ascii')
        for reference_idx, key_name in enumerate(hdf5_data_grp):

            self.assertAlmostEqual(
                pdf_maker_obj.reference_redshift_array[reference_idx],
                hdf5_data_grp[key_name].attrs['redshift'], places=6)
            self.assertEqual(
                pdf_maker_obj.reference_region_array[reference_idx],
                hdf5_data_grp[key_name].attrs['region'])

            self.assertAlmostEqual(
                pdf_maker_obj.reference_area_array[reference_idx],
                hdf5_data_grp[
                    '%s/%s' % (key_name,
                               self.dummy_args.pair_scale_name)
                    ].attrs['area'])
            self.assertEqual(
                pdf_maker_obj.reference_resolution_array[reference_idx],
                hdf5_data_grp[
                    '%s/%s' % (key_name,
                               self.dummy_args.pair_scale_name)
                    ].attrs['bin_resolution'])

            ref_row_list = tmp_output_file.readline().split(' ')
            ref_unkn = np.float32(ref_row_list[0])
            if ref_unkn > 0:
                self.assertAlmostEqual(
                    pdf_maker_obj.reference_unknown_array[reference_idx] /
                    ref_unkn - 1, 0.0, places=6)
            else:
                self.assertAlmostEqual(
                    pdf_maker_obj.reference_unknown_array[reference_idx],
                    ref_unkn, places=6)
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

    def test_compute_region_densities_and_weights(self):

        test_rand_ration = [0.0101507, 0.01101003, 0.01151311, 0.01081756,
                            0.01281959, 0.01212378, 0.0128798, 0.01246496]
        test_ave_weight = np.ones_like(test_rand_ration)

        cosmos_data = fits.getdata(
            'data/COSMOS_iband_2009_radecidstomp_regionzp_best.fits')
        cosmos_zp_cut = cosmos_data[np.logical_and(
            cosmos_data.zp_best > 0.3, cosmos_data.zp_best <= 0.5)]

        open_hdf5_data_file = h5py.File(self.dummy_args.input_pair_hdf5_file)
        hdf5_data_grp = open_hdf5_data_file['data']

        id_array, rand_ratio, weight_array, ave_weight = \
            pdf_maker_utils._compute_region_densities_and_weights(
                cosmos_zp_cut, hdf5_data_grp, self.dummy_args)

        for rand, ave, test_rand, test_ave in \
                zip(rand_ratio, ave_weight, test_rand_ration,
                    test_ave_weight):
            self.assertAlmostEqual(rand, test_rand)
            self.assertAlmostEqual(ave, test_ave)

        open_hdf5_data_file.close()

    def test_load_pair_data(self):

        open_hdf5_data_file = h5py.File(self.dummy_args.input_pair_hdf5_file)
        hdf5_data_grp = open_hdf5_data_file['data']

        key_list = hdf5_data_grp.keys()[:100]

        input_tuple = (self.dummy_args.input_pair_hdf5_file,
                       self.dummy_args.pair_scale_name,
                       key_list)
        output_list = pdf_maker_utils._load_pair_data(input_tuple)

        for output, key in zip(output_list, key_list):
            self.assertEqual(
                len(output['ids']),
                len(hdf5_data_grp[
                        '%s/%s' %
                        (key, self.dummy_args.pair_scale_name)]['ids']))
            self.assertEqual(
                len(output['dist_weights']),
                len(hdf5_data_grp[
                        '%s/%s' %
                        (key,
                         self.dummy_args.pair_scale_name)]['dist_weights']))

        open_hdf5_data_file.close()


if __name__ == '__main__':

    unittest.main()
