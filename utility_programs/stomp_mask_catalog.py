
import argparse
from astropy.io import fits
import numpy as np
import stomp

"""
Utility program for taking an input fits catalog with RA and DEC and masking the
objects to the input stomp map geometry. This is useful for getting the 
random-random normalization term to line up with the data file used.
"""


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stomp_map', required = True,
                        type = str, help = 'Name of the STOMP map defining the '
                        'geometry on the sky for the target and unknown '
                        'samples.')
    parser.add_argument('--n_regions', default = None,
                        type = int, help = 'Number of sub resgions to break up '
                        'the stomp map into for bootstrap/jackknifing. It is '
                        'recommended that the region size be no smaller than '
                        'the max scale requested in degrees.')
    parser.add_argument('--region_column_name', default = 'stomp_region',
                        type = str, help = 'Name of the column to write the '
                        'region_id to.')
    parser.add_argument('--input_fits_file', required = True,
                        type = str, help = 'Name of the fits catalog to load '
                        'and test against the stomp map geometry.')
    parser.add_argument('--output_fits_file', required = True,
                        type = str, help = 'Name of the fits file to write the '
                        'resultant masked data to.')
    parser.add_argument('--ra_name', default = 'ALPHA_J2000',
                        type = str, help = 'Name of ra column in unknown, '
                        'photometric fits file')
    parser.add_argument('--dec_name', default = 'DELTA_J2000',
                        type = str, help = 'Name of dec column in '
                        'unknown, photometric fits file')
    parser.add_argument('--fits_hdu_number', default = 1,
                        type = int, help = 'Number of the fits hdu to load.')
    args = parser.parse_args()
    
    ### Load the fits catalog
    hdu = fits.open(args.input_fits_file)
    data = hdu[args.fits_hdu_number].data
    
    ### Load the stomp map
    stomp_map = stomp.Map(args.stomp_map)
    if args.n_regions is not None:
        stomp_map.InitializeRegions(args.n_regions)
        print("%i regions initialzied at %i resolution" %
              (stomp_map.NRegion(), stomp_map.RegionResolution()))
    
    ### Create a empty mask of the objects considered from the input_fits_file
    mask = np.zeros(data.shape[0], dtype = np.bool)
    if args.n_regions is not None:
        region_array = np.empty(data.shape[0], dtype = np.uint32)
    print("Masking...")
    for idx, obj in enumerate(data):
        if idx % (data.shape[0] / 10) == 0:
            print("\tObject #%i..." % idx)
        tmp_ang = stomp.AngularCoordinate(obj[args.ra_name], obj[args.dec_name],
                                          stomp.AngularCoordinate.Equatorial)
        ### Test the current catalog object and see if it is contained in the
        ### stomp map geometry. Store the result.
        mask[idx] = stomp_map.Contains(tmp_ang)
        if args.n_regions is not None and mask[idx]:
            region_array[idx] = stomp_map.FindRegion(tmp_ang)
    print("\tkept %i / %i" % (data[mask].shape[0], data.shape[0]))
    
    ### Write file to disk and close the currently open fits file.
    col_list = []
    for idx in xrange(len(data.names)):
        if (args.n_regions is not None and
            data.names[idx] == args.region_column_name):
            continue
        col_list.append(fits.Column(name = data.names[idx],
                                    format = data.formats[idx],
                                    array = data[data.names[idx]][mask]))
    if args.n_regions is not None:
        col_list.append(fits.Column(name = args.region_column_name,
                                    format = 'I',
                                    array = region_array[mask]))
    out_tbhdu = fits.BinTableHDU.from_columns(col_list)
    out_tbhdu.writeto(args.output_fits_file, clobber = True)
    hdu.close()
    ### Done!