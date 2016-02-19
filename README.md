# The-wiZZ


INTRODUCTION
------------

The-wiZZ is a clustering redshift recovery analysis tool designed with ease of
use for end users in mind. Simply put, The-wiZZ returns an estimate of the
redshift distribution of a sample of galaxies with unknown redshifts by
cross-correlating with a targert sample of known redshifts. For futher details
on the method see Schmidt et al. 2013, Menard et al 2013, and
Rahman et al. 2015(ab).

The software is composed of two main parts: a pair finder and a pdf maker. The
pair finder does the inital heavy lifting of spatial pair finding and stores
the indices of all closer pairs around the target objects in an output HDF5
data file. Users then query this datafile using pdf_maker.py and the indices of
their unknown sample.

REQUIREMENTS
------------

The libary is designed with as little reliance on nonstandard Python libraries
as possible. It is recommended if using The-wiZZ that you utilize the Anaconda
(https://www.continuum.io/downloads) distribution of Python.

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



CONFIGURATION
-------------



TROUBLESHOOTING
---------------

FAQ
---



MAINTAINERS
-----------

Current:
 * Christopher Morison (AIfA) - https://github.com/morriscb