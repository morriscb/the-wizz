FROM jupyter/scipy-notebook:latest

MAINTAINER Christopher Morrison "morrison.chrisb@gmail.com"

RUN pip install pyarrow && \
    pip install astropy

USER root

WORKDIR /home
RUN git clone https://github.com/morriscb/the-wizz.git
WORKDIR /home/the-wizz
RUN git checkout u/morriscb/python-only && \
    python setup.py install

USER $NB_UID
