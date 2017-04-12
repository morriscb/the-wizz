#! /usr/bin/python

import argparse
from astropy import wcs
from astropy.io import fits
import numpy as np
import stomp

"""
This program allows for the creation of STOMP map from a fits mask assuming that
the masked pixels have positive values greater than the specified cut. It is 
recommended when specifying a resolution that is be roughly twice the area of 
the pixels in the fits file. If the user would like to use a resolution that is
roughly the size of the input image pixels it is recommended that they first 
oversample the fits image pixels before using the software. For reference a
resolution "1" pixel has an area of 88.1474 deg^2 and each resolution decreases
the area by a factor of resolution^2'
"""

def create_exclusion(input_mask, output_map_name, 
                     max_resolution, max_load, mask_value,
                     mask_is_less_than, verbose):
    """
    Function for creating an exclusion mask from the input fits mask. Mismatch
    between STOMP pixels and the fits image pixels could cause area the should
    be masked to be unmasked. By creating the exclusion first and making it
    slightly larger than the input masked region we can guarantee that all
    masked area is properly masked.
    Args:
        input_mask: string name for fits image mask to load.
        output_map_name: string output name of the exclusion mask. If None then
            no map is written.
        max_resolution: resolution at which to attempt to create the stomp
            exclusion mask.
        max_load: int number of pixels to load before dumping them into the
            STOMP map.
        verbose: bool print check statements.
    Returns:
        a stomp.Map object
    """
    try:
        mask = fits.getdata(input_mask)
        ### Load the WCS from the input fits file.
    except ValueError:
        print "ERROR: Failed to load", input_mask, "exiting"
        return None
    hdulist = fits.open(input_mask)
    w = wcs.WCS(hdulist[0].header)
    
    if mask_is_less_than:
        max_pix = len(mask[mask < mask_value])
    else:
        max_pix = len(mask[mask > mask_value])
    # print "Total:", max_pix
    print "Max Pix:", max_pix
    pix_vect = stomp.PixelVector()
    
    tot_pix = 0
    output_map = stomp.Map()
    for idx1 in xrange(mask.shape[0]):
        for idx2 in xrange(mask.shape[1]):
            if not mask_is_less_than and mask[idx1, idx2] <= 0:
                continue
            elif mask_is_less_than and mask[idx1, idx2] >= 0:
                continue
            #print idx1, idx2
            wcs_point = w.wcs_pix2world(np.array([[idx2 + 1, idx1 + 1]],
                                                   np.float_), 1)
            # print idx1, idx2, wcs_point
            # print wcs_point
            ang = stomp.AngularCoordinate(wcs_point[0,0], wcs_point[0,1],
                                          stomp.AngularCoordinate.Equatorial)
            if output_map.Contains(ang):
                continue
            pix_vect.push_back(stomp.Pixel(ang, max_resolution, 1.0))
            if pix_vect.size() >= max_load:
                tmp_map = stomp.Map(pix_vect)
                if verbose:
                    print("Created temp map. Area: %s" %tmp_map.Area())
                output_map.IngestMap(tmp_map)
                tot_pix += pix_vect.size()
                pix_vect.clear()
                del tmp_map
    tmp_map = stomp.Map(pix_vect)
    output_map.IngestMap(tmp_map)
    tot_pix += pix_vect.size()
    pix_vect.clear()
    del tmp_map
            
    print "Final Map:", output_map.Area(), output_map.Size()
    if output_map_name is not None:
        output_map.Write(output_map_name)
    
    return output_map


def create_excluded_map(input_mask, ext_map, output_name, resolution,
                        offset, n_points, counter_clockwise_pixels, verbose):
    """
    Given the input mask, a STOMP map to exclude with, we create the covering
    for the maximum RA and DEC in fits image mask and exclude the area
    specified in the STOMP map creating a output STOMP map that we can then use
    for correlations and other analyses.
    Args:
        input_mask: string name of the input fits image file
        ext_map: stomp.Map object specifying the area to exclude
        output_name: string name of the file to write the resultant STOMP map to
        resolution: int resolution to create the covering STOMP map at.
        offset: int number of pixels to offset from the edge of the image.
        verbose: bool print check statemetns to stdout
    """
    hdu = fits.open(input_mask)
    w = wcs.WCS(hdu[0].header)
    
    
    naxis1_edge_step = ((hdu[0].header['NAXIS1'] - 1 - 2 * offset) /
                        (1. * n_points))
    naxis2_edge_step = ((hdu[0].header['NAXIS2'] - 1 - 2 * offset) /
                        (1. * n_points))
    
    if counter_clockwise_pixels:
        ang_vect = stomp.AngularVector()
        ### South edge:
        for p_idx in xrange(n_points):
            tmp_point = w.wcs_pix2world(
                np.array([[1 + offset + p_idx * naxis1_edge_step,
                           1 + offset]],
                          np.float_), 1)[0]
            ang_vect.push_back(stomp.AngularCoordinate(
            tmp_point[0], tmp_point[1],
            stomp.AngularCoordinate.Equatorial))
        ### West edge:
        for p_idx in xrange(n_points):
            tmp_point = w.wcs_pix2world(
                np.array([[hdu[0].header['NAXIS1'] - offset,
                           1 + offset + p_idx * naxis2_edge_step]],
                          np.float_), 1)[0]
            ang_vect.push_back(stomp.AngularCoordinate(
                tmp_point[0], tmp_point[1],
                stomp.AngularCoordinate.Equatorial))
        ### North edge:
        for p_idx in xrange(n_points):
            tmp_point = w.wcs_pix2world(
                np.array([[hdu[0].header['NAXIS1'] - offset -
                           p_idx * naxis1_edge_step,
                           hdu[0].header['NAXIS2'] - offset]],
                          np.float_), 1)[0]
            ang_vect.push_back(stomp.AngularCoordinate(
                tmp_point[0], tmp_point[1],
                stomp.AngularCoordinate.Equatorial))
        ### East edge:
        for p_idx in xrange(n_points):
            tmp_point = w.wcs_pix2world(
                np.array([[1 + offset,
                           hdu[0].header['NAXIS2'] - offset -
                           p_idx * naxis2_edge_step]],
                         np.float_), 1)[0]
            ang_vect.push_back(stomp.AngularCoordinate(
                tmp_point[0], tmp_point[1],
                stomp.AngularCoordinate.Equatorial))
    else:
        ang_vect = stomp.AngularVector()
        ### South edge:
        for p_idx in xrange(n_points):
            tmp_point = w.wcs_pix2world(
                np.array([[1 + offset,
                           1 + offset + p_idx * naxis2_edge_step]],
                          np.float_), 1)[0]
            ang_vect.push_back(stomp.AngularCoordinate(
            tmp_point[0], tmp_point[1],
            stomp.AngularCoordinate.Equatorial))
        ### West edge:
        for p_idx in xrange(n_points):
            tmp_point = w.wcs_pix2world(
                np.array([[ 1 + offset + p_idx * naxis1_edge_step,
                           hdu[0].header['NAXIS2'] - offset]],
                          np.float_), 1)[0]
            ang_vect.push_back(stomp.AngularCoordinate(
                tmp_point[0], tmp_point[1],
                stomp.AngularCoordinate.Equatorial))
        ### North edge:
        for p_idx in xrange(n_points):
            tmp_point = w.wcs_pix2world(
                np.array([[hdu[0].header['NAXIS1'] - offset ,
                           hdu[0].header['NAXIS2'] - offset -
                           p_idx * naxis2_edge_step]],
                          np.float_), 1)[0]
            ang_vect.push_back(stomp.AngularCoordinate(
                tmp_point[0], tmp_point[1],
                stomp.AngularCoordinate.Equatorial))
        ### East edge:
        for p_idx in xrange(n_points):
            tmp_point = w.wcs_pix2world(
                np.array([[hdu[0].header['NAXIS1'] - offset -
                           p_idx * naxis1_edge_step,
                           1 + offset]],
                         np.float_), 1)[0]
            ang_vect.push_back(stomp.AngularCoordinate(
                tmp_point[0], tmp_point[1],
                stomp.AngularCoordinate.Equatorial))

    bound = stomp.PolygonBound(ang_vect)
    print("Max Area: %.8f" % bound.Area())
    output_stomp_map = stomp.Map(bound, 1.0, resolution, verbose)
    output_stomp_map.ExcludeMap(ext_map)
    print("Final Map Area: %.8f" % output_stomp_map.Area())
    output_stomp_map.Write(output_name)
    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fits_mask', required=True,
                        type=str, help='Input fits mask file. Masked data is '
                        'assumed to have a value of value > mask_value.')
    parser.add_argument('--mask_value', default=0.0,
                        type=np.float_, help='Value above which to mask. '
                        'The excluded area is thus area with a value of '
                        'value > mask_value')
    parser.add_argument('--mask_is_less_than', action='store_true',
                        help='Instead of the default '
                        'behavior, we also allow for masks that are less than '
                        'the mask value to be masked.')
    parser.add_argument('--output_map_name', required=True,
                        type=str, help='Name of output, masked STOMP map '
                        'file.')
    parser.add_argument('--input_exclusion_name', default=None,
                        type=str, help='Name of input exclusion map. If '
                        'None the code creates the exclusion from the '
                        'input_fits_mask. ')
    parser.add_argument('--output_exclusion_name', default=None,
                        type=str, help='Name of output STOMP map to exclude '
                        'from the area. If the value is none, the intermediate '
                        'exclusion map is not written.')
    parser.add_argument('--resolution', default=2048,
                        type=int, help='Resolution at which to '
                        'pixelate the input mask file. Maximum allowed '
                        'resolution is 32768. For reference a resolution 1 '
                        'pixel has an area of 88.1474 deg^2 and each '
                        'resolution decreases the area by a factor of '
                        'resolution^2')
    parser.add_argument('--offset', default=1,
                        type=int, help='Value to offset from the '
                        'edge of the image if image coordiantes.')
    parser.add_argument('--n_points', default=1,
                        type=int, help='Number of points to sample along. '
                        'The edge of the survey boundry. This is helpful for '
                        'High laditude fields. A warning may occur stating the '
                        'area of the polygon created is negative. As long as '
                        'the map returns successful creation in a reasonable '
                        'time, the created stomp map will still be correct.')
    parser.add_argument('--counter_clockwise_pixels', action='store_true',
                        help='Specify which direction the code '
                        'moves around the edge of the image to create the '
                        'map. By default, the code will work if increasing '
                        'x and increasing y are in the direction of increasing '
                        'RA and DEC respectivily.')
    parser.add_argument('--max_load', default=1000000,
                        type=int, help='Number of image pixels to load '
                        'before dumping them into the map. This is a trick to '
                        'reduce the ammount of memory needed for creating the '
                        'exclusion map. It also makes the map creation faster.')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Output full verbosity for stomp')
    args = parser.parse_args()
    
    if args.input_exclusion_name is None:
        ext_map = create_exclusion(args.input_fits_mask,
                                   args.output_exclusion_name,
                                   args.resolution, args.max_load,
                                   args.mask_value, args.mask_is_less_than,
                                   args.verbose)
    else:
        ext_map = stomp.Map(args.input_exclusion_name)
    create_excluded_map(args.input_fits_mask, ext_map, args.output_map_name,
                        args.resolution, args.offset, args.n_points,
                        args.counter_clockwise_pixels, args.verbose)
    