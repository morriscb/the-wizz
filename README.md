# the-wizz


INTRODUCTION
------------

the-wizz is a clustering redshift estimating tool designed with ease of use for
end users in mind. Simply put, the-wizz allows for clustering redshift estimates
for any photometric, unknown sample in a survey by storing all close pairs
between the unknown sample and a target, reference sample into a data file.
Users then query this data file with their specific selection and produce a
clusting redshift. For further details on the method see Schmidt et al. 2013,
Menard et al 2013, Rahman et al. 2015(ab), and Morrison et al 2016.

The software is composed of two main parts: a pair finder and a pdf maker.
pair_finder.py does the initial heavy lifting of spatial pair finding and stores
the indices of all closer pairs around the reference objects in an output HDF5
data file. Users then query this data file using pdf_maker.py and the indices of
their unknown sample, producing an output clustering-z.

CITING the-wizz
---------------

Papers utilizing the-wizz should provide a link back to this repository. It is
also requested that users cite
[Morrison et al. 2016](http://adsabs.harvard.edu/abs/2016arXiv160909085M). The
other cites mentioned at the start of this README are highly recommended as
citations as well.

REQUIREMENTS
------------

The library is designed with as little reliance on nonstandard  libraries
as possible. It is recommended if using the-wizz that you utilize the Anaconda
(https://www.continuum.io/downloads) distribution of .

pdf_maker.py requirements:

    astropy (http://www.astropy.org/)
    h5py (http://www.h5py.org/)
    numpy (http://www.numpy.org/)
    scipy (http://www.scipy.org/)

pair_maker.py requirements:

    (as above)
    astro-stomp (https://github.com/ryanscranton/astro-stomp)

INSTALLATION
------------

the-wizz is available as a [docker](https://www.docker.com/) image for easy
deployment. [This image](https://hub.docker.com/r/morriscb/the-wizz/) can be
deployed with the following docker command

    docker pull morriscb/the-wizz

Tutorials using the docker image of the-wizz will be available soon.

------------------------------------------------------------------------------

the-wizz can also be installed from git using the command

    git clone https://github.com/morriscb/the-wizz.git <dir_name>

Running pdf_maker.py or pair_maker.py from the created directory will properly
run the code.

DEMOS
-----

Demos for installing and configuring astro-stomp, creating masks, running
pair_maker.py, and running pdf_maker.py to create a clustering-z estimate can
be found on the wiki page

https://github.com/morriscb/the-wizz/wiki

These demos use public data from the COSMOS and zCOSMOS surveys to guide users
through the process of running the-wizz in full.

TROUBLESHOOTING
---------------

The format of the-wizz's data file has changed as of 4/12/17. Data files made
with the code before this will not work with the current code and must be
converted.

pdf_maker.py
------------

pdf_maker.py should always come with two files, a fits file containing all of
the photometric, unknown galaxies masked to the area of the refrence and unknown
sample overlaps and the HDF5 data file containing all of the close pairs between
the unknown and reference sample. Users should select their galaxy sample (e.g.
galaxies of a certain color, properites, photometric redshift) from the fits
catalog and match the IDs into the HDF5 file using pdf_maker.py

For mulit-epoch surveys spanning multiple pointings, large area surveys, or
combining different surveys it is recommended to set the
uknown_stomp_region_name flag to the appropriate column name. Having this
flag not set for such surveys will likely result in unexpected results.

Higher signal to noise is achieved with the flag use_inverse_weighting set.
Setting this mode is recommended.

Care should be taken that the correct index column is being used. It should be
the same as that stored in the pair HDF5 file.

pair_maker.py
-------------

This portion of the code is for experts only. The majority of end users will
only use pdf_maker.py.

This part of the code should be used by surveys interested in using the-wizz as
their redshift clustering-z code. This portion of the code creates the HDF5 data
file of all close pairs that is used in pdf_maker.py. It is recommeneded that
surveys use their full, photometric catalog masked to the same area as the
reference catalog used.

The code uses the spatial pixelization library STOMP for all of it's pair
finding. Those unfamiliar with this libary are recommened to have a look at the
source code header files at https://github.com/ryanscranton/astro-stomp.

To use pair finder, one much first create a file describing the usmasked area of
the survey, in STOMP this is called a Map. Two utility functions are available
to create these Maps, stomp_adapt_map.py and stomp_map_from_fits.py.
stomp_map_from_fits.py takes in a fits image descripbing the mask and creates an
aproximation of the unmasked area. stomp_adapt_map.py should be used when no
fits mask or weight map exists. It attempts to intuit the mask from an input
catalog of objects. Descriptions of how to use the code are contained in the
respective  files. It is possible to use STOMP to create a mask from
complex polygons (e.g. ds9 regions, mangle) using code available in the library.
Look to the STOMP::Map and STOMP::Geometry classes for more information.

stomp_mask_catalog.py allows one to mask an input fits catalog to the geomety of
a STOMP Map. It is an extremely useful program as it allows for the creation of
a catalog with the same geometry as that used in pair_maker.py. A catalog
produced from stomp_mask_catalog.py allows the end user to select their sample
from a catalog that has the same geometry as used in the pair finder and thus
all of the average densities will correct. It also allows the ablity to store
the same regionation as used in the pair finder enabling the use of the
"unknown_stomp_region_name" flag in pdf_maker.py. This flag is extremely import
for inhomengious surveys.

The number of regions is an extremely important choice when running the-wizz.
The number of regions requested should be a compromise between smoothing the
scale of individual pointings/systematics and allowing for the largest physical
scale requested. For instance if you have 1 sq deg. pointings, you'll want to
try to have the regions you request be at most 1 sq deg to smooth pointing to
pointing variations from survey stragegy/data quality variation.

Using unquie indices for the target, reference objects can allow one to combine
the data files produced after the fact, enabling simple paralization for the
pair creation process. Make sure to sum together the total number of randoms
through.

For large unknown catalogs where large is not that large (>100k) it is
sufficient to create at most 10 times the number of randoms.

FAQ
---

Q: Why the-wizz?
A: the-wizz is designed to take the hard work (i.e. pair finding) and separate
it from the science (clustering redshift estimation) allowing end users to
simply select from a photometric, unknown catalog and match indices into the
code and procude high significance clustering redshift estimates without the
need to re-run a pair finding/correlation technique every time. the-wizz can be
thought of as creating a value added catalog for your survey much like a
photometric redshift code.

Q: I'd like to use the-wizz for my survey, how do I credit you?
A: There is a helpful section just above here about citing the-wizz.

Q: I'm having trouble using STOMP, why don't you use [insert favorite
correlation code].
A: STOMP has a ton of convince functions that make this code possible and is
 wrapped to make it even more convienent. If you would like to use a
library other than STOMP in the code that can be done assuming that the majority
of the functionality is retained. Feel free to contact the maintainers if you
run into problems creatinging inherited methods/modifying the code.

Q: My install of 2.7 is not working with the-wizz, can you fix it?
A: The recommended install of  is Anaconda
(https://www.continuum.io/downloads) while I would like to be comptable with
everyone's different installs I can not guarentee full compadiblity. If you
run into an issue with the-wizz please add it to the GitHub issue tracker and I
will attempt to address it.

Q: My clustering-z doesn't look right, it either has an incorrect normalization or
just looks weird.
A: Make sure that the photometric, unknown catalog is masked to the same
geometry as was used to create the HDF5 data file using pair_maker.py. For
multi-epoch surveys, usage of the flag "unknown_stomp_region_name" in
pdf_maker.py should be considered a default mode.

Q: Does your software account for galaxy bias in the clustering-z.
A: No. The clustering-zs produced by the-wizz contain no correction for either
the bias of the unknown sample or the reference sample. Users will have their
preference on which method of bias mitigation is "correct" and are encouraged to
use which ever method suits their needs. the-wizz will can be used easily enough
with redshift, color pre-selections as in Rahman et al. 2016(ab).

Q: You call the outputs "PDFs" but they are not normalized to one and sometimes
have negative values, what gives?
A: Clustering-zs return an estimate of the over-density as a function of
redshift that is then normalized into a PDF. Because it is an estimate that is
measured from data, noise can cause points in the clustering-z to be negative
but consistent with zero. There are also some unknown galaxy selections that can
anti-correlate with the reference sample at given redshifts. This problem could
be solved with an appropreate weighting scheme on the reference catalog. The
PDFs returned by the-wizz are unomalized because everyone has their favorite
technique to do this (e.g. spline integral, trapzoid sum, rectangular sum). The
choice is left to the user.

MAINTAINERS
-----------

Current:
 * Christopher Morison (University of Washington) - https://github.com/morriscb