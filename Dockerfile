FROM ubuntu:18.04

RUN apt-get update && \
    apt-get upgrade && \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y python3.6 \
                       python3-pip \
                       libpcl-dev \
                       software-properties-common

RUN  add-apt-repository -y ppa:sweptlaser/python3-pcl
RUN apt install -y python3-pcl

RUN pip3 install -U jupyter matplotlib pandas numpy plotly h5py sklearn torch tqdm

