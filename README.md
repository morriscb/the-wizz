# The-wiZZ


INTRODUCTION
------------

The-wiZZ is a clustering redshift recovery analysis tool designed with ease of
use for end users in mind. Simply put, The-wiZZ returns an estimate of the
redshift distribution of a sample of galaxies with unknown redshifts by
cross-correlating with a target sample of known redshifts. For further details
on the method see Schmidt et al. 2013, Menard et al 2013, and
Rahman et al. 2015(ab).

The software is composed of two main parts: a pair finder and a pdf maker. The
pair finder does the initial heavy lifting of spatial pair finding and stores
the indices of all closer pairs around the target objects in an output HDF5
data file. Users then query this data file using pdf_maker.py and the indices of
their unknown sample.

REQUIREMENTS
------------

The library is designed with as little reliance on nonstandard Python libraries
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

Currently The-wiZZ can be installed from git using the command

    git clone https://github.com/morriscb/The-wiZZ.git <dir_name>

Running pdf_maker.py or pair_maker.py from the created directory will properly
run the code.

TROUBLESHOOTING
---------------

pdf_maker.py
------------

For mulit-epoch surveys spanning multiple pointings, large area surveys, or
combining different surveys it is recommended to set the
uknown_stomp_region_name flag to the appropriate column name. Having this
flag not set for such surveys will likely result in unexpected results.

Higher signal to noise is achieved with the flag use_inverse_weighting set.
Setting this mode is recommended.

Care should be taken that the correct index column is being used. It should be
the same as that stored in the pair HDF5 file.

Requesting larger scales for the correlation requires much more computing power.
If the code is taking a significant amount of time (~30 minutes) per sample,
increase the number of processes. (n_processes)

pair_maker.py
-------------

This portion of the code is for experts only. The majority of users will only
use pdf_maker.py.

FAQ
---



MAINTAINERS
-----------

Current:
 * Christopher Morison (AIfA) - https://github.com/morriscb