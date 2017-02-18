FROM zachdeibert/autotools

MAINTAINER Christopher Morrison "morrison.chrisb@gmail.com"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install git -y && \
    apt-get install zlib1g-dev -y && apt-get install python-dev -y && \
    apt-get install swig -y && apt-get install python-pip -y &&\
    python -m pip install --upgrade pip && \
    pip install --user numpy scipy h5py astropy

# Install the libraries and build astro-stomp
WORKDIR /home
RUN git clone https://github.com/morriscb/astro-stomp.git
WORKDIR /home/astro-stomp
RUN ./autogen.sh && ./configure && make; exit 0 && make install; exit 0

# Build the python wrappers for astro-stomp
WORKDIR /home/astro-stomp/python
RUN python runswig.py && python setup.py install

# Clone and install The-wiZZ
WORKDIR /home
RUN git clone https://github.com/morriscb/The-wiZZ.git
WORKDIR /home/The-wiZZ
RUN chmod u+x pair_maker.py pdf_maker.py \
    utility_programs/stomp_adapt_map.py \
    utility_programs/stomp_map_from_fits.py \
    utility_programs/stomp_mask_catalog.py
ENV PATH /home/The-wiZZ:/home/The-wiZZ/utility_programs:$PATH