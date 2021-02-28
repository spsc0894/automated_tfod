FROM ubuntu:18.04

USER root
RUN apt-get update && apt-get -y upgrade && apt-get autoremove

RUN apt-get install -y --no-install-recommends \
        build-essential \
        xdg-utils \
        apt-utils \
        cpio \
        curl \
        vim \
        git \
        lsb-release \
        pciutils \
        python3.5 \
        python3-pip \
        libgflags-dev \
        libboost-dev \
        libboost-log-dev \
        cmake \
        libx11-dev \
        libssl-dev \
        locales \
        libjpeg8-dev \
        libopenblas-dev \
        gnupg2 \
        protobuf-compiler \
        python-dev \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
        sudo 

RUN pip3 install wheel
RUN pip3 install --upgrade pip
ADD . /app
WORKDIR /app/Tensorflow/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
WORKDIR /app/cocoapi/PythonAPI
RUN apt-get install -y --no-install-recommends python python-pip
RUN python -m pip install setuptools numpy Cython
RUN make
RUN cp -r pycocotools /app/Tensorflow/models/research/
WORKDIR /app/Tensorflow/models/research
RUN cp object_detection/packages/tf2/setup.py .
RUN python3 -m pip install .
RUN python3 object_detection/builders/model_builder_tf2_test.py
RUN mkdir -p /app/Tensorflow/workspace/training_demo/images
WORKDIR /app/Tensorflow/workspace/training_demo/images
RUN cp /app/ds/* .
RUN mkdir -p /app/Tensorflow/scripts/preprocessing
RUN wget https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/d0e545609c5f7f49f39abc7b6a38cec3/partition_dataset.py
RUN python3 partition_dataset.py -x -i /app/Tensorflow/workspace/training_demo/images -r 0.1
RUN mkdir -p /app/Tensorflow/workspace/training_demo/annotations
WORKDIR /app/Tensorflow/workspace/training_demo/annotations
RUN cp /app/label_map.pbtxt .
RUN mkdir -p /app/Tensorflow/scripts/preprocessing

RUN pip install pandas
WORKDIR /app/Tensorflow/scripts/preprocessing
RUN wget https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py

RUN python3 generate_tfrecord.py -x /app/Tensorflow/workspace/training_demo/images/train -l /app/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o /app/Tensorflow/workspace/training_demo/annotations/train.record

RUN python3 generate_tfrecord.py -x /app/Tensorflow/workspace/training_demo/images/test -l /app/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o /app/Tensorflow/workspace/training_demo/annotations/test.record

RUN mkdir -p /app/Tensorflow/workspace/training_demo/pre-trained-models

WORKDIR /app/Tensorflow/workspace/training_demo/pre-trained-models

RUN wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

RUN tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

RUN mkdir -p /app/Tensorflow/workspace/training_demo/models/my_ssd_resnet50_v1_fpn
WORKDIR /app/Tensorflow/workspace/training_demo/models/my_ssd_resnet50_v1_fpn

RUN cp /app/pipeline.config .

RUN cp /app/Tensorflow/models/research/object_detection/model_main_tf2.py /app/Tensorflow/workspace/training_demo/

WORKDIR /app/Tensorflow/workspace/training_demo/

RUN python3 model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config

RUN cp /app/Tensorflow/models/research/object_detection/exporter_main_v2.py /app/Tensorflow/workspace/training_demo/

WORKDIR /app/Tensorflow/workspace/training_demo/
RUN python3 exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/my_ssd_resnet50_v1_fpn/pipeline.config --trained_checkpoint_dir models/my_ssd_resnet50_v1_fpn/ --output_directory exported-models/my_model

WORKDIR /app

CMD ['python3','test_on_sample.py']


