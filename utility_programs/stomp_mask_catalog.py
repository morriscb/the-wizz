
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
    parser.add_argument('--stomp_map', default = '',
                        type = str, help = 'Name of the STOMP map defining the '
                        'geometry on the sky for the target and unknown '
                        'samples.')
    parser.add_argument('--input_fits_file', default = '',
                        type = str, help = 'Name of the fits catalog to load '
                        'and test against the stomp map geometry.')
    parser.add_argument('--output_fits_file', default = '',
                        type = str, help = 'Name of the fits file to write the '
                        'resultant masked data to.')
    parser.add_argument('--ra_name', default = 'ALPHA_J2000',
                        type = str, help = 'Name of ra column in unknown, '
                        'photometric fits file')
    parser.add_argument('--dec_name', default = 'DELTA_J2000',
                        type = str, help = 'Name of dec column in '
                        'unknown, photometric fits file')
    args = parser.parse_args()
    
    ### Load the fits catalog
    hdu = fits.open(args.input_fits_file)
    data = hdu[1].data
    
    ### Load the stomp map
    stomp_map = stomp.Map(args.stomp_map)
    
    ### Create a empty mask of the objects considered from the input_fits_file
    mask = np.zeros(data.shape[0], type = np.bool)
    for idx, obj in enumerate(data):
        tmp_ang = stomp.AngularCoordinate(obj[args.ra_name], obj[args.dec_name],
                                          stomp.AngularCoordinate.Equatorial)
        ### Test the current catalog object and see if it is contained in the
        ### stomp map geometry. Store the result.
        mask[idx] = stomp_map.Contains(tmp_ang)
    
    ### Write file to disk and close the currently open fits file.
    data[mask].writeto(args.output_fits_file)
    hdu.close()
    ### Done!