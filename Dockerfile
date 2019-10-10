FROM jupyter/scipy-notebook:latest

MAINTAINER Christopher Morrison "morrison.chrisb@gmail.com"

RUN pip install pyarrow && \
    pip install astropy
