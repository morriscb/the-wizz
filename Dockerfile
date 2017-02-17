FROM zachdeibert/autotools

MAINTAINER Christopher Morrison "morrison.chrisb@gmail.com"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install git -y && apt-get install zlib1g-dev -y
RUN apt-get install python-dev -y && apt-get install swig -y
RUN apt-get install python-pip -y
RUN python -m pip install --upgrade pip
RUN pip install --user numpy scipy h5py astropy

# Install the libraries and build astro-stomp
WORKDIR /home
RUN git clone https://github.com/morriscb/astro-stomp.git
WORKDIR /home/astro-stomp
RUN ./autogen.sh
ENV LD_LIBRARY_PATH /usr/local/lib
ENV C_INCLUDE_PATH /usr/local/include
ENV CPLUS_INCLUDE_PATH /usr/local/include
RUN ./configure
RUN make; exit 0
RUN make install; exit 0

# Build the python wrappers for astro-stomp
WORKDIR /home/astro-stomp/python
RUN python runswig.py && python setup.py install

# Clone and install The-wiZZ
WORKDIR /home
RUN git clone https://github.com/morriscb/The-wiZZ.git
WORKDIR /home/The-wiZZ
RUN chmod u+x pair_maker.py
RUN chmod u+x pdf_maker.py
ENV PATH /home/The-wiZZ:$PATH