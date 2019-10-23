# the-wizz


INTRODUCTION
------------

the-wizz is a clustering redshift estimating tool designed with ease of use for
end users in mind. Simply put, the-wizz allows for clustering redshift estimates
for any photometric, unknown sample in a survey by storing all close pairs
between the unknown sample and a target, reference sample into a data file.
Users then query this data file with their specific selection and correlation
scales to produce a clusting redshift. For further details on the method see
Schmidt et al. 2013, Menard et al 2013, Rahman et al. 2015(ab), and
Morrison et al 2017.

The software is composed of three main parts: a pair_finder, pair_collapser and
pdf_maker. pair_finder does the initial heavy lifting of spatial pair
finding and stores the indices and compressed distances of all closer pairs
around the reference objects in an output Parquet data file. Users then query
this data file using pdf_collapser with the indices of their unknown sample,
producing an output densities around each reference object. From there
pdf_maker bins the data in redshift and computes the final clustering-zs.

CITING the-wizz
---------------

Papers utilizing the-wizz should provide a link back to this repository. It is
also requested that users cite
[Morrison et al. 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.467.3576M/abstract). The
other cites mentioned at the start of this README are highly recommended as
citations as well.

REQUIREMENTS
------------

The library is designed with as little reliance on nonstandard libraries
as possible.

    astropy (http://www.astropy.org/)
    numpy (http://www.numpy.org/)
    pandas (https://pandas.pydata.org/)
    pyarrow (https://arrow.apache.org/docs/python/install.html)
    scipy (http://www.scipy.org/)

INSTALLATION
------------

To install the-wizz, simply clone the repository and run the setup.py script.

    git clone https://github.com/morriscb/the-wizz
    cd the-wizz
    python setup.py install

A pypi install will be coming in the future.

From there you can run the unittests:

    python tests/test_pair_maker.py
    python tests/test_pair_collapser.py
    python tests/test_pdf_maker.py

DOCKER
======

the-wizz is available as a [docker](https://www.docker.com/) image for easy
deployment. [This image](https://hub.docker.com/r/morriscb/the-wizz/) can be
deployed with the following docker command

    docker pull morriscb/the-wizz

Tutorials using the docker image of the-wizz will be available soon.

DEMOS
-----

Jupyter notebook demos coming soon.

MAINTAINERS
-----------

Current:
 * Christopher Morison (University of Washington) - https://github.com/morriscb
