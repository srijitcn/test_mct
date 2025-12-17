FROM mosaicml/pytorch:2.7.0_cu126-python3.12-ubuntu22.04

RUN apt-get update \
  && apt-get install git -y \
  && mkdir /mnist \
  && mkdir /mnist/datasets \
  && mkdir /checkpoints \
  && mkdir /workdir

COPY requirements_1.txt /workdir/.
COPY requirements_2.txt /workdir/.
 
RUN pip install -r /workdir/requirements_1.txt
RUN pip install -r /workdir/requirements_2.txt


