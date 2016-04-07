
import argparse
from astropy.io import fits
import stomp


"""
Utility program for intuiting the geometry of a survey from an input catalog of
objects. The code transforms each object's position into a STOMP pixel at a
fixed resolution. The pixels are joined together to form a rough area of the
survey. The resolution should be set such that at least one galaxy occupies the
pixel's area. For reference a resolution 1 pixel has an area of 88.1474 deg^2
and each resolution decreases the resolution by (resolution ^ 2).
"""

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fits_file', required = True,
                        type = str, help = 'Name of the fits catalog to load '
                        'and test against the stomp map geometry.')
    parser.add_argument('--output_stomp_map', required = True,
                        type = str, help = 'Name of the ascii file to write '
                        'resultant stomp map to.')
    parser.add_argument('--ra_name', default = 'ALPHA_J2000',
                        type = str, help = 'Name of ra column in unknown, '
                        'photometric fits file')
    parser.add_argument('--dec_name', default = 'DELTA_J2000',
                        type = str, help = 'Name of dec column in '
                        'unknown, photometric fits file')
    parser.add_argument('--resolution', default = 2048,
                        type = int, help = 'Resolution to attempt to pixelize '
                        'the map at.')
    parser.add_argument('--n_object_batch', default = 100000,
                        type = int, help = 'number of objects to run before '
                        'storing into the resultant map.')
    args = parser.parse_args()
    
    hdu = fits.open(args.input_fits_file)
    data = hdu[1].data
    ra_min = data[args.ra_name].min()
    ra_max = data[args.ra_name].max()
    dec_min = data[args.dec_name].min()
    dec_max = data[args.dec_name].max()
    
    print("Creating bounding area...")
    latlon_bound = stomp.LatLonBound(dec_min, dec_max, ra_min, ra_max,
                                     stomp.AngularCoordinate.Equatorial)
    print("\tMax possible Area: %.8f deg^2" % latlon_bound.Area())
    
    print("Creating STOMP map...")
    output_stomp_map = stomp.Map()
    pix_vect = stomp.PixelVector()
    for obj_idx, obj in enumerate(data):
        if (obj_idx + 1) % args.n_object_batch == 0:
            output_stomp_map.IngestMap(pix_vect) 
            pix_vect = stomp.PixelVector()
        tmp_ang = stomp.AngularCoordinate(obj[args.ra_name], obj[args.dec_name],
                                          stomp.AngularCoordinate.Equatorial)
        pix_vect.push_back(stomp.Pixel(tmp_ang, args.resolution, 1.0))
    output_stomp_map.IngestMap(pix_vect)
    
    print("\tFinal Map Area: %.8f deg ^2; Ratio vs bounding box: %.8f" %
          (output_stomp_map.Area(),
           output_stomp_map.Area() / latlon_bound.Area()))
    
    output_stomp_map.Write(args.output_stomp_map)